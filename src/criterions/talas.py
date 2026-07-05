import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F
from src.criterions.utils import count_clean_text_tokens, get_hidden_text, get_hidden_text_vision, pooling

import os


class Talas(nn.Module):
    def __init__(self, args):
        super(Talas, self).__init__()
        self.args = args
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
        self.kd_weight = args.kd_weight
    
    def _dist_gather_tensor(self, t: torch.Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def cosine_loss(self, student_embeddings, teacher_embeddings):
        cos_sim = F.cosine_similarity(student_embeddings, teacher_embeddings, dim=-1)
        cos_sim_loss = 1 - cos_sim
        return cos_sim_loss.mean()

    def structure_loss(self, student_embeddings, teacher_embeddings):
        student_embeddings = F.normalize(student_embeddings, p=2, dim=-1)
        teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=-1)

        student_similarity = student_embeddings @ student_embeddings.transpose(-1, -2)
        teacher_similarity = teacher_embeddings @ teacher_embeddings.transpose(-1, -2)

        loss = F.mse_loss(student_similarity, teacher_similarity)

        return loss

    def get_teacher_representations(self, caching_dir, input_data):
        encoded_dirs = input_data['encoded_dir'] # list
        encoded_dirs = [os.path.join(caching_dir, encoded_dir) for encoded_dir in encoded_dirs]

        teacher_qry_reps = torch.stack([torch.load(os.path.join(encoded_dir, 'qry.pt')) for encoded_dir in encoded_dirs])
        teacher_pos_reps = torch.stack([torch.load(os.path.join(encoded_dir, 'pos.pt')) for encoded_dir in encoded_dirs])

        return teacher_qry_reps, teacher_pos_reps

    def forward(self, model_wrapper, input_data):
        student_model = model_wrapper.model
        projectors = model_wrapper.projectors        

        student_qry_input = input_data['qry']
        student_pos_input = input_data['pos']

        caching_dir = self.args.caching_dir
        
        batch_size = student_qry_input['input_ids'].size(0)

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output

        device = student_qry_reps.device

        teacher_qry_reps, teacher_pos_reps = self.get_teacher_representations(caching_dir, input_data)

        teacher_qry_reps = teacher_qry_reps.to(device)
        teacher_pos_reps = teacher_pos_reps.to(device)
        
        if self.world_size > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
            # all_teacher_qry_reps = self._dist_gather_tensor(teacher_qry_reps)
            # all_teacher_pos_reps = self._dist_gather_tensor(teacher_pos_reps)
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps
            # all_teacher_qry_reps = teacher_qry_reps
            # all_teacher_pos_reps = teacher_pos_reps
            
        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / model_wrapper.temperature, target)

        num_stu_layer = len(student_qry_hidden_states)
        tamd = 0.0

        for proj_idx, i in enumerate(range(num_stu_layer - self.args.num_projectors, 
                       num_stu_layer)):

            last_stu_qry_hidden_state = pooling(student_qry_hidden_states[i], 
                                                student_qry_input['attention_mask'], 
                                                mode='eos',
                                                normalize=False)
            student_qry_proj = projectors[proj_idx](last_stu_qry_hidden_state)
            tamd += self.cosine_loss(student_qry_proj, teacher_qry_reps)

            last_stu_pos_hidden_state = pooling(student_pos_hidden_states[i], 
                                                student_pos_input['attention_mask'], 
                                                mode='eos',
                                                normalize=False)
            student_pos_proj = projectors[proj_idx](last_stu_pos_hidden_state)
            tamd += self.cosine_loss(student_pos_proj, teacher_pos_reps)
        
        tamd /= (2 * self.args.num_projectors)

        lasd = 0.0
        for i in range(num_stu_layer - 1 - self.args.num_self_kd_layers,
                       num_stu_layer - 1):
            
            last_stu_qry_hidden_state_i = pooling(student_qry_hidden_states[i], 
                                                student_qry_input['attention_mask'], 
                                                mode='eos',
                                                normalize=False)
            last_stu_qry_hidden_state_i1 = pooling(student_qry_hidden_states[i+1], 
                                                student_qry_input['attention_mask'], 
                                                mode='eos',
                                                normalize=False)
            lasd += self.structure_loss(last_stu_qry_hidden_state_i, last_stu_qry_hidden_state_i1)


            last_stu_pos_hidden_state_i = pooling(student_pos_hidden_states[i],
                                                student_pos_input['attention_mask'],
                                                mode='eos',
                                                normalize=False)
            last_stu_pos_hidden_state_i1 = pooling(student_pos_hidden_states[i+1],
                                                student_pos_input['attention_mask'],
                                                mode='eos',
                                                normalize=False)
            lasd += self.structure_loss(last_stu_pos_hidden_state_i, last_stu_pos_hidden_state_i1)

        lasd /= (2 * self.args.num_self_kd_layers)
        
        loss_distill = tamd + lasd

        loss = contrastive_loss + self.kd_weight * loss_distill

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': loss_distill
        }
        