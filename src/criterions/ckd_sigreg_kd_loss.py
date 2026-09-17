import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CKDSigRegLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.kd_loss_weight = args.kd_weight

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

    def _dist_gather_tensor(self, t: Tensor):
        if self.world_size == 1:
            return t

        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t

        return torch.cat(all_tensors, dim=0)

    def sigreg(self, x: torch.Tensor, num_slices: int = 128) -> torch.Tensor:
        device = x.device

        if self.process_rank == 0:
            projection_seed = random.randint(0, 2**63 - 1)
        else:
            projection_seed = 0

        if self.world_size > 1:
            seed_tensor = torch.tensor(projection_seed, dtype=torch.int64, device=device)
            dist.broadcast(seed_tensor, src=0)
            projection_seed = seed_tensor.item()

        g = torch.Generator(device=device)
        g.manual_seed(projection_seed)

        A = torch.randn(x.size(1), num_slices, generator=g, device=device, dtype=x.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)

        t = torch.linspace(-5, 5, 17, device=device, dtype=x.dtype)
        exp_f = torch.exp(-0.5 * t.square())

        x_t = (x @ A).unsqueeze(-1) * t
        ecf = torch.exp(1j * x_t).mean(dim=0)

        if self.world_size > 1:
            dist.all_reduce(ecf, op=dist.ReduceOp.SUM)
            ecf = ecf / self.world_size

        err = (ecf - exp_f).abs().square().mul(exp_f)
        global_batch_size = x.size(0) * self.world_size
        sigreg_per_slice = torch.trapezoid(err, t, dim=1) * global_batch_size

        return sigreg_per_slice.mean()

    def ckd_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        student_diff = student.unsqueeze(1) - student.unsqueeze(0)
        teacher_diff = teacher.unsqueeze(1) - teacher.unsqueeze(0)

        return F.mse_loss(student_diff, teacher_diff)

    def forward(self, distiller, input_data):
        student_model = distiller.student
        teacher_model = distiller.teacher
        projectors = distiller.projectors

        student_qry_input = input_data["student_inputs"]["qry"]
        student_pos_input = input_data["student_inputs"]["pos"]
        teacher_qry_input = input_data["teacher_inputs"]["qry"]
        teacher_pos_input = input_data["teacher_inputs"]["pos"]

        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
            teacher_qry_reps, _, _, _ = teacher_qry_output
            teacher_pos_reps, _, _, _ = teacher_pos_output

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, _, _, _ = student_qry_output
        student_pos_reps, _, _, _ = student_pos_output

        # =====================================================
        # InfoNCE
        # =====================================================
        all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
        all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)

        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))

        contrastive_loss = F.cross_entropy(scores / distiller.temperature, target)

        # =====================================================
        # Project teacher to student embedding dimension
        # =====================================================
        projected_teacher_qry_reps = projectors["t2s"](teacher_qry_reps)
        projected_teacher_pos_reps = projectors["t2s"](teacher_pos_reps)

        all_projected_teacher_qry_reps = self._dist_gather_tensor(projected_teacher_qry_reps)
        all_projected_teacher_pos_reps = self._dist_gather_tensor(projected_teacher_pos_reps)

        # =====================================================
        # Comparative Knowledge Distillation
        #
        # z = [query embeddings, positive embeddings]
        #
        # L_CKD = MSE(
        #     z_s[i] - z_s[j],
        #     z_t[i] - z_t[j]
        # )
        # =====================================================
        student_ckd_reps = torch.cat([all_student_qry_reps, all_student_pos_reps], dim=0)
        teacher_ckd_reps = torch.cat([all_projected_teacher_qry_reps, all_projected_teacher_pos_reps], dim=0)

        kd_loss = self.ckd_loss(student_ckd_reps, teacher_ckd_reps)

        # =====================================================
        # SIGReg
        # =====================================================
        sigreg_qry_loss = self.sigreg(student_qry_reps)
        sigreg_pos_loss = self.sigreg(student_pos_reps)
        sigreg_loss = sigreg_qry_loss + sigreg_pos_loss

        # =====================================================
        # Total
        # =====================================================
        loss = contrastive_loss + self.kd_loss_weight * kd_loss + self.args.sigreg_weight * sigreg_loss

        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
            "sigreg_loss": sigreg_loss,
            "sigreg_qry_loss": sigreg_qry_loss,
            "sigreg_pos_loss": sigreg_pos_loss,
        }