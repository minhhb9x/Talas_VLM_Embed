import json
from src.single_wrapper import SingleWrapper, SingleCollator, SingleDataset
from src.arguments import DataArguments, MTEBArguments, TrainingArguments, ModelArguments
from src import model
from src.utils import print_rank, print_master
from src.criterions import build_criterion
import time 
import os
import sys
from tqdm import tqdm 
import math
# import wandb 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, Optimizer

from accelerate import Accelerator
from huggingface_hub import HfApi, HfFolder, Repository, create_repo
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser
from transformers.integrations import HfDeepSpeedConfig
# Todo
import random
import numpy as np


def should_use_muon(name, param):
    if not param.requires_grad or param.ndim != 2:
        return False
    name = name.lower()
    adam_keywords = ("embed", "embedding", "lm_head", "head", "norm", "ln")
    return not any(keyword in name for keyword in adam_keywords)


def is_adamw_projector(name):
    return (
        name.startswith("projectors.")
        or ".mm_projector." in name
        or name.endswith(".mm_projector.weight")
        or name.endswith(".mm_projector.bias")
        or ".multi_modal_projector." in name
        or name.endswith(".multi_modal_projector.weight")
        or name.endswith(".multi_modal_projector.bias")
    )


class CombinedOptimizer(Optimizer):
    def __init__(self, optimizers, adamw_defaults):
        self.optimizers = optimizers
        self.adamw_defaults = adamw_defaults
        params = []
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                params.extend(group["params"])
        self._initializing = True
        super().__init__(params, defaults={})
        self._initializing = False
        self._refresh_param_groups()

    def _refresh_param_groups(self):
        self.param_groups = [
            group
            for optimizer in self.optimizers
            for group in optimizer.param_groups
        ]

    def add_param_group(self, param_group):
        if self._initializing:
            return super().add_param_group(param_group)

        adamw_optimizer = next(
            (
                optimizer
                for optimizer in self.optimizers
                if optimizer.param_groups and not optimizer.param_groups[0].get("use_muon", False)
            ),
            None,
        )
        param_group.setdefault("lr", self.adamw_defaults["lr"])
        param_group.setdefault("weight_decay", self.adamw_defaults["weight_decay"])
        param_group.setdefault("betas", self.adamw_defaults["betas"])
        param_group.setdefault("eps", self.adamw_defaults["eps"])
        param_group.setdefault("name", "adamw")
        param_group.setdefault("use_muon", False)
        if adamw_optimizer is None:
            adamw_optimizer = AdamW([param_group])
            adamw_optimizer.param_groups[0]["name"] = "adamw"
            adamw_optimizer.param_groups[0]["use_muon"] = False
            self.optimizers.insert(0, adamw_optimizer)
        else:
            adamw_optimizer.add_param_group(param_group)
        self._refresh_param_groups()

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        for optimizer in self.optimizers:
            step_loss = optimizer.step(closure=closure if loss is None else None)
            if loss is None:
                loss = step_loss
        return loss

    def state_dict(self):
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers]
        }

    def load_state_dict(self, state_dict):
        for optimizer, optimizer_state in zip(self.optimizers, state_dict["optimizers"]):
            optimizer.load_state_dict(optimizer_state)
        self._refresh_param_groups()


def seed_everything(seed: int, rank: int = 0):
    seed = seed + rank  # quan trọng trong DDP

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Nếu bạn muốn deterministic (chậm hơn, đôi khi lỗi với một số ops)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Bắt buộc với một số ops CUDA mới (matmul, conv...)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_optimizer_params(model, training_args):
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if p.requires_grad]},
    ]

    return optimizer_grouped_parameters

def get_optimizer(model, training_args):
    while isinstance(model, DDP):
        model = model.module

    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError(
            "torch.optim.Muon is not available. Please install torch>=2.9."
        )

    muon_params = []
    adamw_params = []
    projector_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_adamw_projector(name):
            projector_params.append(param)
        elif should_use_muon(name, param):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    adamw_defaults = {
        "lr": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    optimizers = []
    adamw_groups = []
    if adamw_params:
        adamw_groups.append({
            "params": adamw_params,
            "lr": training_args.learning_rate,
            "weight_decay": training_args.weight_decay,
            "name": "adamw",
            "use_muon": False,
        })
    if projector_params:
        model_args = getattr(model, "model_args", None)
        projector_lr = getattr(model_args, "projector_lr", None) or training_args.learning_rate
        adamw_groups.append({
            "params": projector_params,
            "lr": projector_lr,
            "weight_decay": training_args.weight_decay,
            "name": "projector_adamw",
            "use_muon": False,
        })
    if adamw_groups:
        adamw_optimizer = AdamW(adamw_groups, **adamw_defaults)
        optimizers.append(adamw_optimizer)
    if muon_params:
        muon_kwargs = {}
        if training_args.muon_lr is not None:
            muon_kwargs["lr"] = training_args.muon_lr
        if training_args.muon_weight_decay is not None:
            muon_kwargs["weight_decay"] = training_args.muon_weight_decay
        muon_optimizer = torch.optim.Muon(
            sorted(muon_params, key=lambda p: p.numel(), reverse=True),
            **muon_kwargs,
        )
        muon_optimizer.param_groups[0]["name"] = "muon"
        muon_optimizer.param_groups[0]["use_muon"] = True
        optimizers.append(muon_optimizer)
    if not optimizers:
        raise ValueError("No trainable parameters found for optimizer")

    optimizer = CombinedOptimizer(optimizers, adamw_defaults=adamw_defaults)
    print_rank(f"Muon params: {sum(p.numel() for p in muon_params):,}")
    print_rank(f"AdamW fallback params: {sum(p.numel() for p in adamw_params):,}")
    print_rank(f"Projector AdamW params: {sum(p.numel() for p in projector_params):,}")
    return optimizer

def prepare_dataset(data_args, model_args):
    dataset = SingleDataset(data_args, model_args)
    return dataset

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def to_device(obj, device):
    if obj is None:
        return None
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        result = [to_device(v, device) for v in obj]
        return tuple(result) if isinstance(obj, tuple) else result
    else:
        if hasattr(obj, 'to') and callable(obj.to):
            return obj.to(device)
        return obj

def ddp_setup():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    init_process_group(backend="nccl")

class Trainer:
    def __init__(self, model_wrapper, train_data, optimizer, lr_scheduler, criterion, 
                 model_args, training_args, data_args):
        print_rank("Initializing Trainer...")
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.device = torch.device(f'cuda:{self.gpu_id}')
        self.model_wrapper = model_wrapper.to(self.device)
        self.train_data = train_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        
        self.model_wrapper = DDP(self.model_wrapper, device_ids=[self.gpu_id], find_unused_parameters=True)

        # <--- [THÊM] Logic kiểm tra report_to="wandb"
        # self.use_wandb = False
        # if is_main_process():
        #     # Kiểm tra xem report_to có tồn tại và chứa wandb không
        #     report_to = getattr(training_args, "report_to", [])
        #     if report_to is None: report_to = []
        #     if isinstance(report_to, str):
        #         report_to = [report_to]
            
        #     if "wandb" in report_to:
        #         self.use_wandb = True
    
    def _debug_batch_devices(self, obj, prefix=""):
        if obj is None:
            print(f"{prefix}Value: None")
            return
        
        try:
            if isinstance(obj, torch.Tensor):
                print(f"{prefix}Tensor device: {obj.device}, shape: {obj.shape}")
            elif isinstance(obj, dict):
                if len(obj) == 0:
                    print(f"{prefix}Empty dict")
                for k, v in obj.items():
                    self._debug_batch_devices(v, prefix=f"{prefix}{k}.")
            elif isinstance(obj, (list, tuple)):
                if len(obj) == 0:
                    print(f"{prefix}Empty {type(obj).__name__}")
                for i, v in enumerate(obj):
                    self._debug_batch_devices(v, prefix=f"{prefix}[{i}].")
            else:
                print(f"{prefix}Type: {type(obj).__name__}, Value: {obj}")
        except Exception as e:
            print(f"{prefix}ERROR: {e}")
        
    def run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        losses, contrastive_losses, kd_losses = [], [], []
        kd_rkd_losses, ot_losses, kd_dtw_losses = [], [], []
        kd_mse_losses, kd_penultimate_losses = [], []
        
        # Tính tổng số bước (steps) trong epoch để log step
        steps_per_epoch = len(self.train_data.dataset) // self.training_args.per_device_train_batch_size // self.training_args.gradient_accumulation_steps // dist.get_world_size()

        progress_bar = tqdm(total=steps_per_epoch, 
                            desc=f"Epoch {epoch}",
                            disable=not dist.get_rank() == 0)
        for batch_idx, batch in enumerate(self.train_data):
            batch = to_device(batch, self.device)
            loss_dict = self.model_wrapper(self.criterion, batch)
            loss = loss_dict['loss'] / self.training_args.gradient_accumulation_steps
            kd_loss = loss_dict.get('kd_loss', torch.tensor(0.0))
            contrastive_loss = loss_dict.get('contrastive_loss', torch.tensor(0.0))
            kd_rkd_loss = loss_dict.get('kd_loss_rkd', torch.tensor(0.0))
            ot_loss = loss_dict.get('ot_loss', torch.tensor(0.0))
            kd_dtw_loss = loss_dict.get('kd_loss_dtw', torch.tensor(0.0))
            kd_mse_loss = loss_dict.get('kd_mse_loss', torch.tensor(0.0))
            kd_penultimate_loss = loss_dict.get('kd_penultimate_loss', torch.tensor(0.0))

            losses.append(loss.detach().item() * self.training_args.gradient_accumulation_steps)
            contrastive_losses.append(contrastive_loss.detach().item())
            kd_losses.append(kd_loss.detach().item())
            kd_rkd_losses.append(kd_rkd_loss.detach().item())
            ot_losses.append(ot_loss.detach().item())
            kd_dtw_losses.append(kd_dtw_loss.detach().item())
            kd_mse_losses.append(kd_mse_loss.detach().item())
            kd_penultimate_losses.append(kd_penultimate_loss.detach().item())
            
            batch_loss = sum(losses) / len(losses)
            batch_contrastive_loss = sum(contrastive_losses) / len(contrastive_losses)
            batch_kd_loss = sum(kd_losses) / len(kd_losses)
            batch_kd_rkd_loss = sum(kd_rkd_losses) / len(kd_rkd_losses)
            batch_ot_loss = sum(ot_losses) / len(ot_losses)
            batch_kd_dtw_loss = sum(kd_dtw_losses) / len(kd_dtw_losses)
            batch_kd_loss_mse = sum(kd_mse_losses) / len(kd_mse_losses)
            batch_kd_penultimate_loss = sum(kd_penultimate_losses) / len(kd_penultimate_losses)
            
            loss.backward()
            if (batch_idx + 1) % self.training_args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
                if is_main_process():
                    postfix = {
                        'loss': f"{batch_loss:.4f}",
                        'kd_loss': f"{batch_kd_loss:.4f}",
                        'contrastive_loss': f"{batch_contrastive_loss:.4f}",
                        'kd_rkd_loss': f"{batch_kd_rkd_loss:.4f}",
                        'ot_loss': f"{batch_ot_loss:.4f}",
                        'kd_dtw_loss': f"{batch_kd_dtw_loss:.4f}",
                        'kd_loss_mse': f"{batch_kd_loss_mse:.4f}",
                        'kd_penultimate_loss': f"{batch_kd_penultimate_loss:.4f}",
                    }
                    for group_idx, group in enumerate(self.optimizer.param_groups):
                        group_name = group.get("name", f"group_{group_idx}")
                        lr_key = f"{group_name}_lr"
                        if lr_key in postfix:
                            lr_key = f"{group_name}_{group_idx}_lr"
                        postfix[lr_key] = f"{group['lr']:.6f}"
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(1)

                    # <--- [THÊM] Log metrics vào wandb
                    # if self.use_wandb:
                    #     # Log loss trung bình (cumulative average) hoặc loss tức thời (instant)
                    #     # Ở đây mình log loss trung bình tích lũy giống như progress bar
                    #     wandb.log({
                    #         "train/loss": batch_loss,
                    #         "train/kd_loss": batch_kd_loss,
                    #         "train/contrastive_loss": batch_contrastive_loss,
                    #         "train/kd_rkd_loss": batch_kd_rkd_loss,
                    #         "train/ot_loss": batch_ot_loss,
                    #         "train/kd_dtw_loss": batch_kd_dtw_loss,
                    #         "train/kd_loss_mse": batch_kd_loss_mse,
                    #         "train/kd_penultimate_loss": batch_kd_penultimate_loss,
                    #         "train/learning_rate": current_lr,
                    #         "train/epoch": epoch + ((batch_idx + 1) / self.training_args.gradient_accumulation_steps) / steps_per_epoch
                    #     })
                
            torch.cuda.empty_cache()
        progress_bar.close()
        
    def train(self):
        # <--- [THÊM] Khởi tạo wandb run
        # if self.use_wandb:
           
        #     all_config = {}
        #     if self.model_args: all_config.update(vars(self.model_args))
        #     if self.data_args: all_config.update(vars(self.data_args))
        #     if self.training_args: all_config.update(vars(self.training_args))

        #     wandb.init(
        #         project="VLM_Embed_distill",
        #         config=all_config,
        #         reinit=True
        #     )

        for epoch in range(self.training_args.num_train_epochs):
            self.run_epoch(epoch)
            if is_main_process() and self.training_args.save_strategy == "epoch":
                ckpt_dir = os.path.join(self.training_args.output_dir, f"checkpoint-epoch-{epoch}")
                projector_dir = os.path.join(ckpt_dir, "mm_projector.pth")
                os.makedirs(ckpt_dir, exist_ok=True)
                
                model = self.model_wrapper.module.model
                model.encoder.save_pretrained(ckpt_dir)
                torch.save(model.encoder.model.model.mm_projector.state_dict(), projector_dir)
                model_config = AutoConfig.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                if model_config:
                    model_config.save_pretrained(ckpt_dir)
                if tokenizer:
                    tokenizer.save_pretrained(ckpt_dir)
                try:
                    processor = AutoProcessor.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                    if processor:
                        processor.save_pretrained(ckpt_dir)
                except Exception as e:
                    print_rank(f"Warning: Could not save processor: {e}")
                print_rank(f"Saved checkpoint to {ckpt_dir}")

        if is_main_process():
            final_ckpt_dir = os.path.join(self.training_args.output_dir, f"checkpoint-final")
            projector_dir =  os.path.join(final_ckpt_dir, "mm_projector.pth")
            os.makedirs(final_ckpt_dir, exist_ok=True)
            model = self.model_wrapper.module.model
            model.encoder.save_pretrained(final_ckpt_dir)
            torch.save(model.encoder.model.model.mm_projector.state_dict(), projector_dir)
            model_config = AutoConfig.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
            tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
            if model_config:
                model_config.save_pretrained(final_ckpt_dir)
            if tokenizer:
                tokenizer.save_pretrained(final_ckpt_dir)
            try:
                processor = AutoProcessor.from_pretrained(self.model_args.model_name) if self.model_args.model_name else None
                if processor:
                    processor.save_pretrained(final_ckpt_dir)
            except Exception as e:
                print_rank(f"Warning: Could not save processor: {e}")
            print_rank(f"Saved final model to {final_ckpt_dir}")
            
            # if self.use_wandb:
            #     wandb.finish()
                
def main():
    for arg in sys.argv:
        if arg.startswith("--local_rank"):
            local_rank = int(arg.split("=")[-1])
            sys.argv.remove(arg)
            sys.argv.append(f"--local_rank")
            sys.argv.append(f"{local_rank}")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    rank = dist.get_rank()
    # seed_everything(training_args.seed, rank=rank) 
    
    model_wrapper = SingleWrapper(model_args, training_args)
    train_dataset = prepare_dataset(data_args, model_args)
    dist_sampler = DistributedSampler(train_dataset, shuffle=True, seed=training_args.seed)
    for n, p in model_wrapper.named_parameters():
        if p.requires_grad:  # thường chỉ là LoRA
            p.data = p.data.to(torch.bfloat16)
    
    collator = SingleCollator(
        processor=model_wrapper.get_processor(),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=dist_sampler,
        collate_fn=collator,
        drop_last=True,
        pin_memory=False,
    )
    num_trainable_vision = 0
    for n, p in model_wrapper.model.named_parameters():
        if "mm_projector" in n or "multi_modal_projector" in n:
            p.requires_grad = True
        if "lm_head" in n:
            p.requires_grad = False
        if p.requires_grad:
            p.data = p.data.to(torch.bfloat16)
            num_trainable_vision += p.numel()
    print_rank(f"Number of trainable vision parameters: {num_trainable_vision}")
    
    optimizer = get_optimizer(model_wrapper, training_args)
    print(f"Len of train dataset: {len(train_dataloader.dataset)}")
    total_steps = (len(train_dataloader.dataset) // (training_args.per_device_train_batch_size * dist.get_world_size()) // training_args.gradient_accumulation_steps) * training_args.num_train_epochs

    print("Number of trainable parameters:", sum(p.numel() for group in optimizer.param_groups for p in group['params'] if p.requires_grad))

    if training_args.lr_scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps,
            num_training_steps=total_steps,
        )
    elif training_args.lr_scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps,
            num_training_steps=total_steps,
        )
    else:
        from transformers import get_constant_schedule_with_warmup
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_ratio * total_steps,
        )
    criterion = build_criterion(training_args)
    trainer = Trainer(model_wrapper, train_dataloader, optimizer, lr_scheduler, criterion, 
                      model_args, training_args, data_args)
    trainer.train()
    
if __name__ == "__main__":
    ddp_setup()
    main()
    destroy_process_group()
