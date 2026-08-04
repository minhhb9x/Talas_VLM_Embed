import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F


CACHE_ROOT = Path("caching")
MODEL_DIRS = {
    "teacher": CACHE_ROOT / "B3_Qwen2_2B_cls",
    "student": CACHE_ROOT / "FastVLM-0.5B_base_8",
}
INPUT_FILES = {
    "query": "qry.pt",
    "target": "pos.pt",
}


def list_datapoints(model_dir: Path) -> set[str]:
    datapoints = set()
    for qry_path in model_dir.glob("*/*/qry.pt"):
        datapoint_dir = qry_path.parent
        if (datapoint_dir / "pos.pt").is_file():
            datapoints.add(datapoint_dir.relative_to(model_dir).as_posix())
    return datapoints


def sample_datapoints(num_samples: int, seed: int | None) -> list[str]:
    common = set.intersection(*(list_datapoints(path) for path in MODEL_DIRS.values()))
    if num_samples > len(common):
        raise ValueError(f"Requested {num_samples}, but only {len(common)} datapoints exist.")

    return random.Random(seed).sample(sorted(common), num_samples)


def load_embeddings(model_dir: Path, datapoints: list[str], filename: str) -> torch.Tensor:
    return torch.stack(
        [torch.load(model_dir / datapoint / filename, map_location="cpu") for datapoint in datapoints]
    ).float()


def embedding_stats(embeddings: torch.Tensor) -> dict[str, float]:
    x = embeddings.float()
    xn = F.normalize(x, dim=-1)
    raw_norm = x.norm(dim=-1)
    raw_var = x.var(dim=0, unbiased=False)
    norm_var = xn.var(dim=0, unbiased=False)

    return {
        "raw_norm_mean": raw_norm.mean().item(),
        "raw_norm_std": raw_norm.std(unbiased=False).item(),
        "raw_var_mean": raw_var.mean().item(),
        "raw_var_sum": raw_var.sum().item(),
        "norm_var_mean": norm_var.mean().item(),
        "norm_var_sum": norm_var.sum().item(),
        "normalized_mean_vector_norm": xn.mean(dim=0).norm().item(),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    datapoints = sample_datapoints(args.num_samples, args.seed)

    print(f"Embedding stats (num_samples={len(datapoints)}, seed={args.seed})")
    for role, model_dir in MODEL_DIRS.items():
        print(f"\n{role.upper()}")
        for input_name, filename in INPUT_FILES.items():
            embeddings = load_embeddings(model_dir, datapoints, filename)
            stats = embedding_stats(embeddings)
            print(f"  {input_name} shape={tuple(embeddings.shape)}")
            print(f"    raw_norm_mean              {stats['raw_norm_mean']:12.8f}")
            print(f"    raw_norm_std               {stats['raw_norm_std']:12.8f}")
            print(f"    raw_var_mean               {stats['raw_var_mean']:12.8f}")
            print(f"    raw_var_sum                {stats['raw_var_sum']:12.8f}")
            print(f"    norm_var_mean              {stats['norm_var_mean']:12.8f}")
            print(f"    norm_var_sum               {stats['norm_var_sum']:12.8f}")
            print(
                f"    normalized_mean_vector_norm "
                f"{stats['normalized_mean_vector_norm']:12.8f}"
            )
