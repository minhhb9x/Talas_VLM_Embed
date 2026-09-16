import argparse
import os

import torch
import torch.nn.functional as F


def compute_effective_rank(
    hidden_state: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    x = hidden_state.float()
    n = x.size(0)

    s = torch.linalg.svdvals(x) / torch.sqrt(
        torch.tensor(n, device=x.device, dtype=x.dtype)
    )

    eigvals = s.square()
    prob = eigvals.clamp_min(eps) / eigvals.sum().clamp_min(eps)

    entropy = -(prob * torch.log(prob)).sum()

    return torch.exp(entropy) / n


def load_hidden_data(
    pt_path: str,
    normalize: bool = False,
):
    obj = torch.load(pt_path, map_location="cpu")

    num_image_tokens = int(obj.get("num_image_tokens", 0))
    num_text_tokens = int(obj.get("num_text_tokens", 0))

    if num_image_tokens <= 0:
        return None, None

    # [num_layers, num_valid_tokens, hidden_dim]
    hidden_state = obj["hidden_state"].float()

    # =========================================================
    # Last token of last hidden layer
    #
    # hidden_state[-1]     : [num_valid_tokens, D]
    # hidden_state[-1, -1] : [D]
    # =========================================================
    last_hidden_last_token = hidden_state[-1, -1, :].clone()

    # =========================================================
    # Image tokens
    # =========================================================
    last_image_token = bool(obj.get("last_image_token", False))

    if last_image_token:
        # detect has eos or not
        has_eos_id = bool(obj.get("has_eos_id", False))

        if has_eos_id:
            res = hidden_state[
                :,
                num_text_tokens:num_text_tokens - 1 + num_image_tokens,
                :,
            ]
        else:
            res = hidden_state[
                :,
                num_text_tokens:num_text_tokens + num_image_tokens,
                :,
            ]
    else:
        res = hidden_state[:, :num_image_tokens, :]

    if normalize:
        res = F.normalize(res, p=2, dim=-1)

    return res, last_hidden_last_token


def compute_per_sample_layer_eranks(
    image_hidden_layers: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute effective rank independently for each layer of one image.

    Input:
        [num_layers, num_image_tokens, hidden_dim]

    Output:
        [num_layers]
    """
    layer_eranks = []

    for layer_hidden in image_hidden_layers:
        erank = compute_effective_rank(layer_hidden.to(device))
        layer_eranks.append(erank.cpu())

    return torch.stack(layer_eranks, dim=0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pt_dir",
        default="infer/B3_Qwen2_2B_low/ImageNet-1K/query",
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=49)

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    per_sample_layer_eranks = []
    image_hidden_samples = []

    # =========================================================
    # NEW:
    # Store last token from the last hidden layer for each sample
    #
    # Each element: [D]
    # After stacking: [num_samples, D]
    # =========================================================
    last_hidden_last_tokens = []

    loaded_files = []

    # ---------------------------------------------------------
    # Load samples
    # ---------------------------------------------------------
    for idx in range(args.start_idx, args.end_idx + 1):

        pt_path = os.path.join(
            args.pt_dir,
            f"{idx:08d}.pt",
        )

        if not os.path.exists(pt_path):
            print(f"Skip missing file: {pt_path}")
            continue

        image_hidden_layers, last_hidden_last_token = load_hidden_data(
            pt_path,
            args.normalize,
        )

        if image_hidden_layers is None:
            print(f"Skip no-image file: {pt_path}")
            continue

        per_sample_layer_eranks.append(
            compute_per_sample_layer_eranks(
                image_hidden_layers,
                device,
            )
        )

        image_hidden_samples.append(image_hidden_layers)

        # [D]
        last_hidden_last_tokens.append(
            last_hidden_last_token
        )

        loaded_files.append(pt_path)

    if not per_sample_layer_eranks:
        raise RuntimeError("No image tokens loaded.")

    # =========================================================
    # Per-sample image-token eRank
    # =========================================================

    # [num_samples, num_layers]
    per_sample_layer_eranks = torch.stack(
        per_sample_layer_eranks,
        dim=0,
    )

    # ---------------------------------------------------------
    # Average per-sample effective rank
    # ---------------------------------------------------------
    avg_per_sample_layer_eranks = per_sample_layer_eranks.mean(
        dim=0
    )

    std_per_sample_layer_eranks = per_sample_layer_eranks.std(
        dim=0,
        unbiased=False,
    )

    # =========================================================
    # NEW:
    # Last token from last hidden state
    #
    # [B, D]
    # =========================================================
    last_hidden_last_tokens = torch.stack(
        last_hidden_last_tokens,
        dim=0,
    )

    # ---------------------------------------------------------
    # 1. Raw last-token effective rank
    # ---------------------------------------------------------
    last_token_erank_raw = compute_effective_rank(
        last_hidden_last_tokens.to(device)
    ).cpu()

    # ---------------------------------------------------------
    # 2. L2-normalized last-token effective rank
    #
    # Normalize each sample independently:
    # [B, D]
    # ---------------------------------------------------------
    normalized_last_hidden_last_tokens = F.normalize(
        last_hidden_last_tokens,
        p=2,
        dim=-1,
    )

    last_token_erank_norm = compute_effective_rank(
        normalized_last_hidden_last_tokens.to(device)
    ).cpu()

    # =========================================================
    # Dataset-level effective ranks for image tokens
    # =========================================================
    all_token_layer_eranks = []
    mean_pooled_layer_eranks = []

    num_layers = image_hidden_samples[0].size(0)

    for layer_idx in range(num_layers):

        # =====================================================
        # 1. All image tokens across all images
        #
        # image 1: [N1, D]
        # image 2: [N2, D]
        # ...
        #
        # concatenate ->
        # [N1 + N2 + ..., D]
        # =====================================================
        all_image_tokens = torch.cat(
            [
                image_hidden[layer_idx]
                for image_hidden in image_hidden_samples
            ],
            dim=0,
        ).to(device)

        all_token_erank = compute_effective_rank(
            all_image_tokens
        ).cpu()

        all_token_layer_eranks.append(all_token_erank)

        # =====================================================
        # 2. Mean-pool tokens within each image first
        #
        # image 1: [N1, D] -> [D]
        # image 2: [N2, D] -> [D]
        # ...
        #
        # stack ->
        # [num_images, D]
        # =====================================================
        mean_pooled_images = torch.stack(
            [
                image_hidden[layer_idx].mean(dim=0)
                for image_hidden in image_hidden_samples
            ],
            dim=0,
        ).to(device)

        mean_pooled_erank = compute_effective_rank(
            mean_pooled_images
        ).cpu()

        mean_pooled_layer_eranks.append(
            mean_pooled_erank
        )

    all_token_layer_eranks = torch.stack(
        all_token_layer_eranks,
        dim=0,
    )

    mean_pooled_layer_eranks = torch.stack(
        mean_pooled_layer_eranks,
        dim=0,
    )

    # =========================================================
    # Print results
    # =========================================================
    print(f"Loaded files: {len(loaded_files)}")

    print(
        "Per-sample image effective rank shape: "
        f"{tuple(per_sample_layer_eranks.shape)}"
    )

    print(
        "Last-hidden last-token batch shape: "
        f"{tuple(last_hidden_last_tokens.shape)}"
    )

    print("\nImage effective rank per layer:")

    for (
        layer_idx,
        mean_erank,
        std_erank,
        all_token_erank,
        mean_pooled_erank,
    ) in zip(
        range(num_layers),
        avg_per_sample_layer_eranks,
        std_per_sample_layer_eranks,
        all_token_layer_eranks,
        mean_pooled_layer_eranks,
    ):
        print(
            f"  layer {layer_idx:02d}: "
            f"per_sample_mean={mean_erank.item():.6f}, "
            f"per_sample_std={std_erank.item():.6f}, "
            f"all_token_erank={all_token_erank.item():.6f}, "
            f"mean_pooled_erank={mean_pooled_erank.item():.6f}"
        )

    print(
        "\nLast layer per-sample mean effective rank: "
        f"{avg_per_sample_layer_eranks[-1].item():.6f}"
    )

    print(
        "Last layer all-token effective rank: "
        f"{all_token_layer_eranks[-1].item():.6f}"
    )

    print(
        "Last layer mean-pooled effective rank: "
        f"{mean_pooled_layer_eranks[-1].item():.6f}"
    )

    # =========================================================
    # NEW RESULTS
    # =========================================================
    print("\nLast hidden state - last token effective rank:")

    print(
        f"  raw:        {last_token_erank_raw.item():.6f}"
    )

    print(
        f"  normalized: {last_token_erank_norm.item():.6f}"
    )


if __name__ == "__main__":
    main()