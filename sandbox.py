import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F


torch.set_printoptions(precision=4, sci_mode=False)

CACHE_ROOT = Path("caching")
MODEL_DIRS = {
    "b3": CACHE_ROOT / "B3_Qwen2_2B_cls",
    "fastvlm": CACHE_ROOT / "FastVLM-0.5B_base_8",
}


def list_cached_datapoints(model_dir: Path) -> set[str]:
    """Return datapoint dirs relative to model_dir that contain qry.pt and pos.pt."""
    datapoints = set()

    for qry_path in model_dir.glob("*/*/qry.pt"):
        datapoint_dir = qry_path.parent
        if (datapoint_dir / "pos.pt").is_file():
            datapoints.add(datapoint_dir.relative_to(model_dir).as_posix())

    return datapoints


def sample_common_datapoints(
    model_dirs: dict[str, Path],
    num_samples: int,
    seed: int | None = None,
) -> list[str]:
    common_datapoints = None

    for model_dir in model_dirs.values():
        cached_datapoints = list_cached_datapoints(model_dir)
        common_datapoints = (
            cached_datapoints
            if common_datapoints is None
            else common_datapoints & cached_datapoints
        )

    if not common_datapoints:
        raise RuntimeError("No common cached datapoints found across all models.")

    common_datapoints = sorted(common_datapoints)
    if num_samples > len(common_datapoints):
        raise ValueError(
            f"Requested {num_samples} samples, but only "
            f"{len(common_datapoints)} common datapoints are available."
        )

    rng = random.Random(seed)
    return rng.sample(common_datapoints, num_samples)


def load_cache_matrix(model_dir: Path, datapoints: list[str], name: str) -> torch.Tensor:
    reps = [
        torch.load(model_dir / datapoint / name, map_location="cpu")
        for datapoint in datapoints
    ]
    return torch.stack(reps, dim=0)


def load_sampled_cache_matrices(
    num_samples: int = 8,
    seed: int | None = 42,
) -> tuple[list[str], dict[str, dict[str, torch.Tensor]]]:
    datapoints = sample_common_datapoints(MODEL_DIRS, num_samples, seed)

    matrices = {}
    for model_name, model_dir in MODEL_DIRS.items():
        matrices[model_name] = {
            "qry": load_cache_matrix(model_dir, datapoints, "qry.pt"),
            "pos": load_cache_matrix(model_dir, datapoints, "pos.pt"),
        }

    return datapoints, matrices


def cosine_similarity_matrix(
    qry_matrix: torch.Tensor,
    pos_matrix: torch.Tensor,
) -> torch.Tensor:
    if qry_matrix.ndim != 2 or pos_matrix.ndim != 2:
        raise ValueError(
            f"qry_matrix and pos_matrix must be 2D, got "
            f"{tuple(qry_matrix.shape)} and {tuple(pos_matrix.shape)}."
        )
    if qry_matrix.size(-1) != pos_matrix.size(-1):
        raise ValueError(
            f"qry_matrix and pos_matrix must have the same embedding dim, got "
            f"{qry_matrix.size(-1)} and {pos_matrix.size(-1)}."
        )

    qry_matrix = F.normalize(qry_matrix.float(), dim=-1)
    pos_matrix = F.normalize(pos_matrix.float(), dim=-1)
    return qry_matrix @ pos_matrix.T


def cosine_similarity_matrices_by_model(
    matrices: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {
        model_name: cosine_similarity_matrix(model_matrices["qry"], model_matrices["pos"])
        for model_name, model_matrices in matrices.items()
    }


def compute_effective_rank(
    matrix: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {tuple(matrix.shape)}.")

    x = matrix.float()
    n = min(x.size(0), x.size(1))
    s = torch.linalg.svdvals(x) / torch.sqrt(torch.tensor(n, device=x.device))
    eigvals = s * s
    prob = eigvals.clamp(min=eps) / eigvals.sum()
    entropy = -(prob * torch.log(prob)).sum()
    return torch.exp(entropy) / n


def effective_ranks_by_model(
    matrices: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        model_name: {
            "qry": compute_effective_rank(model_matrices["qry"]),
            "pos": compute_effective_rank(model_matrices["pos"]),
            "qry_pos": compute_effective_rank(
                torch.cat([model_matrices["qry"], model_matrices["pos"]], dim=0)
            ),
        }
        for model_name, model_matrices in matrices.items()
    }


def format_matrix(
    matrix: torch.Tensor,
    precision: int = 4,
    width: int = 8,
) -> str:
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {tuple(matrix.shape)}.")

    rows = matrix.detach().cpu().float().tolist()
    return "\n".join(
        "[" + " ".join(f"{value:{width}.{precision}f}" for value in row) + "]"
        for row in rows
    )


def print_matrix(
    name: str,
    matrix: torch.Tensor,
    precision: int = 4,
    width: int = 8,
) -> None:
    print(f"\n{name}:")
    print(format_matrix(matrix, precision=precision, width=width))


def print_effective_ranks(
    eranks: dict[str, dict[str, torch.Tensor]],
) -> None:
    print("\nEffective ranks:")
    for model_name, model_eranks in eranks.items():
        qry_erank = model_eranks["qry"].item()
        pos_erank = model_eranks["pos"].item()
        qry_pos_erank = model_eranks["qry_pos"].item()
        print(
            f"{model_name}: "
            f"qry={qry_erank:.4f}, "
            f"pos={pos_erank:.4f}, "
            f"qry_pos={qry_pos_erank:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sampled_datapoints, sampled_matrices = load_sampled_cache_matrices(
        num_samples=args.num_samples,
        seed=args.seed,
    )

    b3_qry_matrix = sampled_matrices["b3"]["qry"]
    b3_pos_matrix = sampled_matrices["b3"]["pos"]
    fastvlm_qry_matrix = sampled_matrices["fastvlm"]["qry"]
    fastvlm_pos_matrix = sampled_matrices["fastvlm"]["pos"]
    cosine_matrices = cosine_similarity_matrices_by_model(sampled_matrices)
    effective_ranks = effective_ranks_by_model(sampled_matrices)
    b3_cosine_matrix = cosine_matrices["b3"]
    fastvlm_cosine_matrix = cosine_matrices["fastvlm"]

    # print("Sampled datapoints:")
    # for datapoint in sampled_datapoints:
    #     print(f"- {datapoint}")

    # print("\nMatrix shapes:")
    # print(f"b3_qry_matrix: {tuple(b3_qry_matrix.shape)}")
    # print(f"b3_pos_matrix: {tuple(b3_pos_matrix.shape)}")
    # print(f"fastvlm_qry_matrix: {tuple(fastvlm_qry_matrix.shape)}")
    # print(f"fastvlm_pos_matrix: {tuple(fastvlm_pos_matrix.shape)}")

    # print("\nCosine matrix shapes:")
    # print(f"b3_cosine_matrix: {tuple(b3_cosine_matrix.shape)}")
    # print(f"fastvlm_cosine_matrix: {tuple(fastvlm_cosine_matrix.shape)}")

    # print_matrix("B3 query-pos cosine matrix", b3_cosine_matrix)
    # print_matrix("FastVLM query-pos cosine matrix", fastvlm_cosine_matrix)
    print_effective_ranks(effective_ranks)
