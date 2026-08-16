import argparse
import os

import torch


def compute_effective_rank(hidden_state: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    x = hidden_state.float()
    n = x.size(0)
    s = torch.linalg.svdvals(x) / torch.sqrt(torch.tensor(n, device=x.device, dtype=x.dtype))
    eigvals = s.square()
    prob = eigvals.clamp_min(eps) / eigvals.sum().clamp_min(eps)
    entropy = -(prob * torch.log(prob)).sum()
    return torch.exp(entropy) / n


def load_image_tokens(pt_path: str) -> torch.Tensor | None:
    obj = torch.load(pt_path, map_location="cpu")
    num_image_tokens = int(obj.get("num_image_tokens", 0))
    if num_image_tokens <= 0:
        return None

    hidden_state = obj["hidden_state"]
    last_hidden_state = hidden_state[-1]  # [num_valid_tokens, dim]
    return last_hidden_state[:num_image_tokens].float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt_dir",
        default="infer/FastVLM-0.5B_base_16_eos_cls/ImageNet-1K/query",
    )
    parser.add_argument("--start_idx", type=int, default=1)
    parser.add_argument("--end_idx", type=int, default=49)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    image_token_chunks = []
    loaded_files = []

    for idx in range(args.start_idx, args.end_idx + 1):
        pt_path = os.path.join(args.pt_dir, f"{idx:08d}.pt")
        if not os.path.exists(pt_path):
            print(f"Skip missing file: {pt_path}")
            continue

        image_tokens = load_image_tokens(pt_path)
        if image_tokens is None:
            print(f"Skip no-image file: {pt_path}")
            continue

        image_token_chunks.append(image_tokens)
        loaded_files.append(pt_path)

    if not image_token_chunks:
        raise RuntimeError("No image tokens loaded.")

    image_tokens = torch.cat(image_token_chunks, dim=0).to(args.device)
    effective_rank = compute_effective_rank(image_tokens)

    print(f"Loaded files: {len(loaded_files)}")
    print(f"Concatenated image tokens shape: {tuple(image_tokens.shape)}")
    print(f"Image effective rank: {effective_rank.item():.6f}")


if __name__ == "__main__":
    main()
