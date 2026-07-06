import gc
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler, SequentialSampler
from tqdm import tqdm
from transformers import HfArgumentParser

from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor
from src.single_wrapper import SingleCollator, SingleDataset
from src.utils import print_rank


def seed_everything(seed: int, rank: int = 0):
    seed = seed + rank
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_dist()) or dist.get_rank() == 0


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def ddp_setup():
    if "LOCAL_RANK" not in os.environ:
        return
    torch.cuda.set_device(get_local_rank())
    dist.init_process_group(backend="nccl")


def cleanup_dist():
    if is_dist():
        dist.destroy_process_group()


def to_device(obj, device):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj


def first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Unsupported teacher output type: {type(output)}")


def release_memory(device):
    gc.collect()
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def prepare_dataset(data_args, model_args):
    return SingleDataset(data_args, model_args)


class DistributedSequentialSampler(Sampler):
    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.indices = list(range(rank, len(dataset), world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class TeacherCacheRunner:
    def __init__(self, teacher, train_data, training_args):
        self.device = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
        self.teacher = teacher.to(self.device)
        self.teacher.eval()
        self.train_data = train_data
        self.training_args = training_args

    @torch.inference_mode()
    def run_once(self):
        progress_bar = tqdm(
            total=len(self.train_data),
            desc="Teacher cache",
            disable=not is_main_process(),
        )

        for batch_idx, batch in enumerate(self.train_data):
            batch = to_device(batch, self.device)

            qry_reps = first_tensor(self.teacher.encode_input(batch["qry"])).detach().cpu()
            pos_reps = first_tensor(self.teacher.encode_input(batch["pos"])).detach().cpu()

            encoded_dirs = batch.get("encoded_dir")
            if encoded_dirs is None:
                raise KeyError("Batch is missing encoded_dir. Make sure SingleDataset.__getitem__ and SingleCollator return it.")

            for i, (encoded_dir, qry_rep, pos_rep) in enumerate(zip(encoded_dirs, qry_reps, pos_reps)):
                output_qry_dir = os.path.join(self.training_args.output_dir, encoded_dir, "qry.pt")
                output_pos_dir = os.path.join(self.training_args.output_dir, encoded_dir, "pos.pt")
                os.makedirs(os.path.dirname(output_qry_dir), exist_ok=True)
                os.makedirs(os.path.dirname(output_pos_dir), exist_ok=True)
                torch.save(qry_rep.clone(), output_qry_dir)
                torch.save(pos_rep.clone(), output_pos_dir)

            progress_bar.update(1)
            del qry_reps
            del pos_reps
            del batch
            release_memory(self.device)

        progress_bar.close()
        return


def build_dataloader(model_args, data_args, training_args):
    train_dataset = prepare_dataset(data_args, model_args)

    if is_dist():
        sampler = DistributedSequentialSampler(
            train_dataset,
            rank=get_rank(),
            world_size=get_world_size(),
        )
    else:
        sampler = SequentialSampler(train_dataset)

    collator = SingleCollator(
        processor=load_processor(model_args, None),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    return DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=collator,
        drop_last=False,
        pin_memory=False,
    )


def normalize_torchrun_args():
    for arg in list(sys.argv):
        if arg.startswith("--local_rank="):
            local_rank = arg.split("=", 1)[1]
            sys.argv.remove(arg)
            sys.argv.extend(["--local_rank", local_rank])


def main():
    normalize_torchrun_args()
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    seed_everything(training_args.seed, rank=get_rank())

    teacher = MMEBModel.load(model_args, is_trainable=False)
    train_dataloader = build_dataloader(model_args, data_args, training_args)

    print_rank(f"Loaded {len(train_dataloader.dataset)} training samples")
    print_rank(f"Running teacher inference over {len(train_dataloader)} batches")

    runner = TeacherCacheRunner(teacher, train_dataloader, training_args)
    os.makedirs(training_args.output_dir, exist_ok=True)
    runner.run_once()

    
    if is_dist():
        dist.barrier()


if __name__ == "__main__":
    ddp_setup()
    try:
        main()
    finally:
        cleanup_dist()
