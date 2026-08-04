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

    def relation_matrix(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Intra-modal: relation_matrix(x)    -> X X^T
        Inter-modal: relation_matrix(x, y) -> X Y^T
        """
        x = F.normalize(x, dim=-1)

        if y is None:
            y = x
        else:
            y = F.normalize(y, dim=-1)

        return x @ y.T


    # def relation_kd_loss(
    #     self,
    #     student_x: torch.Tensor,
    #     teacher_x: torch.Tensor,
    #     student_y: torch.Tensor | None = None,
    #     teacher_y: torch.Tensor | None = None,
    #     remove_diagonal: bool = False,
    #     temperature: float = 0.02,
    # ) -> torch.Tensor:
    #     if temperature <= 0:
    #         raise ValueError(f"temperature must be positive, got {temperature}")

    #     student_logits = self.relation_matrix(
    #         student_x.float(),
    #         student_y.float() if student_y is not None else None,
    #     )
    #     teacher_logits = self.relation_matrix(
    #         teacher_x.float(),
    #         teacher_y.float() if teacher_y is not None else None,
    #     )

    #     if student_logits.shape != teacher_logits.shape:
    #         raise ValueError(
    #             f"Shape mismatch: student={student_logits.shape}, "
    #             f"teacher={teacher_logits.shape}"
    #         )

    #     diagonal_mask = None
    #     if remove_diagonal:
    #         if student_logits.size(-2) != student_logits.size(-1):
    #             raise ValueError(
    #                 "remove_diagonal=True requires a square relation matrix"
    #             )

    #         diagonal_mask = torch.eye(
    #             student_logits.size(-1),
    #             dtype=torch.bool,
    #             device=student_logits.device,
    #         )

    #         # Phải mask trước softmax để phần tử đường chéo
    #         # không tham gia vào phân phối xác suất.
    #         student_logits = student_logits.masked_fill(
    #             diagonal_mask,
    #             float("-inf"),
    #         )
    #         teacher_logits = teacher_logits.masked_fill(
    #             diagonal_mask,
    #             float("-inf"),
    #         )

    #     student_log_probs = F.log_softmax(
    #         student_logits / temperature,
    #         dim=-1,
    #     )

    #     with torch.no_grad():
    #         teacher_probs = F.softmax(
    #             teacher_logits / temperature,
    #             dim=-1,
    #         )

    #     if diagonal_mask is not None:
    #         # F.kl_div still evaluates masked entries. Avoid 0 * -inf on the
    #         # diagonal, which otherwise produces NaN.
    #         student_log_probs = student_log_probs.masked_fill(diagonal_mask, 0.0)
    #         teacher_probs = teacher_probs.masked_fill(diagonal_mask, 0.0)

    #     return (
    #         F.kl_div(
    #             student_log_probs,
    #             teacher_probs,
    #             reduction="batchmean",
    #         )
    #         * temperature**2
    #     )

    def forward(self, model_wrapper, input_data):
        student_model = model_wrapper.model
        projectors = model_wrapper.projectors        

        student_qry_input = input_data['qry']
        student_pos_input = input_data['pos']
        
        batch_size = student_qry_input['input_ids'].size(0)

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output

        device = student_qry_reps.device

        teacher_qry_reps, teacher_pos_reps = input_data["teacher_qry_rep"], input_data["teacher_pos_rep"]

        teacher_qry_reps = teacher_qry_reps.to(device)
        teacher_pos_reps = teacher_pos_reps.to(device)
        
        if self.world_size > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
            all_teacher_qry_reps = self._dist_gather_tensor(teacher_qry_reps)
            all_teacher_pos_reps = self._dist_gather_tensor(teacher_pos_reps)
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps
            all_teacher_qry_reps = teacher_qry_reps
            all_teacher_pos_reps = teacher_pos_reps
            
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
                                                normalize=True)
            student_qry_proj = projectors[proj_idx](last_stu_qry_hidden_state)
            tamd += self.cosine_loss(student_qry_proj, teacher_qry_reps)

            last_stu_pos_hidden_state = pooling(student_pos_hidden_states[i], 
                                                student_pos_input['attention_mask'], 
                                                mode='eos',
                                                normalize=True)
            student_pos_proj = projectors[proj_idx](last_stu_pos_hidden_state)
            tamd += self.cosine_loss(student_pos_proj, teacher_pos_reps)

        tamd /= (2 * self.args.num_projectors)

        # qq_loss = self.relation_kd_loss(
        #     all_student_qry_reps,
        #     all_teacher_qry_reps,
        #     remove_diagonal=True,
        #     temperature=model_wrapper.temperature
        # )

        # tt_loss = self.relation_kd_loss(
        #     all_student_pos_reps,
        #     all_teacher_pos_reps,
        #     remove_diagonal=True,
        #     temperature=model_wrapper.temperature
        # )

        # qt_loss = self.relation_kd_loss(
        #     all_student_qry_reps,
        #     all_teacher_qry_reps,
        #     all_student_pos_reps,
        #     all_teacher_pos_reps,
        #     temperature=model_wrapper.temperature
        # )

        # tamd = (qq_loss + tt_loss + qt_loss) / 3.0

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
