import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F
from src.criterions.utils import count_clean_text_tokens, get_hidden_text, get_hidden_text_vision, pooling
import random
import os


class TalasJepa(nn.Module):
    def __init__(self, args):
        super(TalasJepa, self).__init__()
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

    def distillcse_kd_loss(
            self, S1, S2,
                T1, T2,
                tau=0.05,):
        """
        Distill teacher similarity distribution over in-batch negatives.

        Student and teacher dimensions do not need to match because
        distillation is applied to pairwise similarity matrices.
        """
        S1 = F.normalize(S1.float(), p=2, dim=-1)

        S2 = F.normalize(S2.float(), p=2, dim=-1)

        T1 = F.normalize(T1.float(), p=2, dim=-1)

        T2 = F.normalize(T2.float(), p=2, dim=-1,)

        s_logits = (S1 @ S2.transpose(0, 1)) / tau

        t_logits = (T1 @ T2.transpose(0, 1)) / tau

        # Positive query-passage pairs are on the diagonal.
        # DistillCSE KD here focuses on the negative distribution.
        mask = torch.eye(s_logits.size(0), device=s_logits.device, dtype=torch.bool,)

        s_logits = s_logits.masked_fill(mask, torch.finfo(s_logits.dtype).min,)

        t_logits = t_logits.masked_fill( mask,torch.finfo(t_logits.dtype).min,)

        teacher_probs = F.softmax(t_logits,dim=1,).detach()

        student_log_probs = F.log_softmax(s_logits,dim=1,)

        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean",)
    
    def sigreg(self, x: torch.Tensor, num_slices: int = 256 ) -> torch.Tensor:
        device = x.device
        # =====================================================
        # 1. Random projection seed
        #
        # Chỉ rank 0 sinh seed.
        # Sau đó broadcast để tất cả GPU dùng cùng seed.
        # =====================================================
        if self.process_rank == 0:
            projection_seed = random.randint(0, 2**63 - 1)
        else:
            projection_seed = 0

        if self.world_size > 1:
            seed_tensor = torch.tensor(projection_seed, dtype=torch.int64, device=device,)
            dist.broadcast(seed_tensor, src=0)
            projection_seed = seed_tensor.item()

        # =====================================================
        # 2. Local generator
        #
        # Không làm thay đổi global torch RNG state.
        # Tất cả GPU có cùng projection_seed -> cùng A.
        # =====================================================
        g = torch.Generator(device=device)
        g.manual_seed(projection_seed)

        A = torch.randn(x.size(1), num_slices, generator=g,  device=device, dtype=x.dtype,)

        A = A / A.norm(p=2, dim=0, keepdim=True, ).clamp_min(1e-12)

        # =====================================================
        # 3. Epps-Pulley statistic
        # =====================================================
        t = torch.linspace(-5, 5, 17, device=device, dtype=x.dtype,)

        exp_f = torch.exp(-0.5 * t.square())

        # x:   [N, K]
        # A:   [K, M]
        # x@A: [N, M]
        # x_t: [N, M, T]
        x_t = (x @ A).unsqueeze(-1) * t

        # [M, T]
        ecf = torch.exp(1j * x_t).mean(dim=0)

        # =====================================================
        # 4. Aggregate across GPUs
        # =====================================================
        if self.world_size > 1:
            dist.all_reduce(ecf, op=dist.ReduceOp.SUM,)
            ecf = ecf / self.world_size

        # =====================================================
        # 5. Weighted L2 distance
        # =====================================================
        err = ((ecf - exp_f).abs().square().mul(exp_f))

        global_batch_size = x.size(0) * self.world_size

        sigreg_per_slice = (torch.trapezoid(err, t, dim=1,) * global_batch_size)

        return sigreg_per_slice.mean()

    def forward(self, model_wrapper, input_data):
        student_model = model_wrapper.model
        student_processor = model_wrapper.get_processor()
        student_tokenizer = student_processor.tokenizer
        projectors = model_wrapper.projectors        

        student_qry_input = input_data['qry']
        student_pos_input = input_data['pos']
        
        batch_size = student_qry_input['input_ids'].size(0)

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output

        device = student_qry_reps.device
        dtype = student_qry_reps.dtype

        teacher_qry, teacher_pos = input_data["teacher_qry_caches"], input_data["teacher_pos_caches"] # list of objects, each object is a tensor of shape [batch_size, hidden_dim]

        teacher_qry_reps = torch.stack([rep['rep'] for rep in teacher_qry], dim=0).to(device, dtype=dtype)
        teacher_pos_reps = torch.stack([rep['rep'] for rep in teacher_pos], dim=0).to(device, dtype=dtype)

        tea_img_qry_reps = torch.stack([rep['mean_last_img_token'] for rep in teacher_qry], 
                                       dim=0,).to(device, dtype=dtype) if teacher_qry[0]['mean_last_img_token'] is not None else None
        tea_img_pos_reps = torch.stack([rep['mean_last_img_token'] for rep in teacher_pos], 
                                       dim=0,).to(device, dtype=dtype) if teacher_pos[0]['mean_last_img_token'] is not None else None
        
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

        kd_simcse = 0.0
        last_stu_qry_hidden_state = pooling(student_qry_hidden_states[-1], 
                                            student_qry_input['attention_mask'], 
                                            mode='eos',
                                            normalize=True)
        last_stu_pos_hidden_state = pooling(student_pos_hidden_states[-1], 
                                            student_pos_input['attention_mask'], 
                                            mode='eos',
                                            normalize=True)
        
        kd_simcse += self.distillcse_kd_loss(last_stu_qry_hidden_state, last_stu_pos_hidden_state, 
                                            teacher_qry_reps, teacher_pos_reps)

        ##################################

        vision_loss = 0.0
        SIGReg = 0.0 
        cur_idx_qry_img = 0
        cur_idx_pos_img = 0

        student_special_ids = torch.tensor(
            list(set(list(student_tokenizer.added_tokens_encoder.values()) + student_tokenizer.all_special_ids) 
                 - set([student_tokenizer.eos_token_id])),
            device=student_qry_input['input_ids'].device,
            dtype=torch.long
        )

        num_student_text_qry_tokens = count_clean_text_tokens(student_qry_input, student_special_ids)
        num_student_text_pos_tokens = count_clean_text_tokens(student_pos_input, student_special_ids)

        if tea_img_qry_reps is not None:
            stu_img_qry_reps = []
            
            for i in range(batch_size):
                if student_qry_image_features is not None and \
                            cur_idx_qry_img < len(student_qry_image_features):
                    img_feat = student_qry_image_features[cur_idx_qry_img]
                    num_vision_token = img_feat.size(0)
                    _, last_stu_img_hidden_state = get_hidden_text_vision(
                        student_qry_hidden_states[-1][i],
                        num_student_text_qry_tokens[i].item(),
                        num_vision_token,
                        student_qry_input['attention_mask'][i]
                    )
                    stu_img_qry_reps.append(last_stu_img_hidden_state.mean(dim=0))
                    cur_idx_qry_img += 1

            if len(stu_img_qry_reps) > 0:
                stu_img_qry_reps = torch.stack(stu_img_qry_reps, dim=0)
                SIGReg = SIGReg + self.sigreg(stu_img_qry_reps, num_slices=256)
                vision_loss += nn.MSELoss()(stu_img_qry_reps, 
                                            projectors['t2s'](tea_img_qry_reps))

        if teacher_pos_reps is not None:
            stu_img_pos_reps = []

            for i in range(batch_size):
                if student_pos_image_features is not None and \
                        cur_idx_pos_img < len(student_pos_image_features):
                    img_feat = student_pos_image_features[cur_idx_pos_img]
                    num_vision_token = img_feat.size(0)
                    _, last_stu_img_hidden_state = get_hidden_text_vision(
                        student_pos_hidden_states[-1][i],
                        num_student_text_pos_tokens[i].item(),
                        num_vision_token,
                        student_pos_input['attention_mask'][i]
                    )
                    stu_img_pos_reps.append(last_stu_img_hidden_state.mean(dim=0))
                    cur_idx_pos_img += 1

            if len(stu_img_pos_reps) > 0:
                stu_img_pos_reps = torch.stack(stu_img_pos_reps, dim=0)
                SIGReg = SIGReg + self.sigreg(stu_img_pos_reps, num_slices=256)
                vision_loss += nn.MSELoss()(stu_img_pos_reps, 
                                            projectors['t2s'](tea_img_pos_reps))

        vision_loss = vision_loss / (len(stu_img_qry_reps) + len(stu_img_pos_reps) + 1e-8)

                

        loss_distill = kd_simcse + vision_loss 

        loss = contrastive_loss + self.kd_weight * loss_distill + 0.05 * SIGReg

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': loss_distill
        }
