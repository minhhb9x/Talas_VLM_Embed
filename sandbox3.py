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

    # =========================================================
    # Full hidden states
    #
    # [num_layers, num_valid_tokens, hidden_dim]
    # =========================================================
    hidden_state = obj["hidden_state"].float()

    # =========================================================
    # Last token from EVERY hidden layer
    #
    # hidden_state[:, -1, :]
    #
    # Shape:
    # [num_layers, hidden_dim]
    # =========================================================
    last_token_all_layers = hidden_state[:, -1, :].clone()

    # =========================================================
    # Extract image tokens
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
        res = hidden_state[
            :,
            :num_image_tokens,
            :,
        ]

    # =========================================================
    # Optional normalization for IMAGE TOKENS
    # =========================================================
    if normalize:
        res = F.normalize(
            res,
            p=2,
            dim=-1,
        )

    return res, last_token_all_layers


def compute_per_sample_layer_eranks(
    image_hidden_layers: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute effective rank independently for each layer
    within ONE sample.

    Input:
        image_hidden_layers:
            [num_layers, num_image_tokens, hidden_dim]

    For every layer:
        [num_image_tokens, hidden_dim]
            ->
        effective rank

    Output:
        [num_layers]
    """

    layer_eranks = []

    for layer_hidden in image_hidden_layers:

        erank = compute_effective_rank(
            layer_hidden.to(device)
        )

        layer_eranks.append(
            erank.cpu()
        )

    return torch.stack(
        layer_eranks,
        dim=0,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pt_dir",
        default="infer/FastVLM-0.5B_talas_1.0_eos_cls/ImageNet-1K/query",
    )

    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end_idx",
        type=int,
        default=49,
    )

    parser.add_argument(
        "--device",
        default="cuda:1",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    # =========================================================
    # Storage
    # =========================================================

    # Per-sample image-token eRank
    #
    # each element:
    # [num_layers]
    per_sample_layer_eranks = []

    # Image hidden states
    #
    # each element:
    # [num_layers, num_image_tokens, D]
    image_hidden_samples = []

    # Last token from every hidden layer
    #
    # each element:
    # [num_layers, D]
    #
    # after stack:
    # [B, num_layers, D]
    last_token_all_layers_samples = []

    loaded_files = []

    # =========================================================
    # Load samples
    # =========================================================
    for idx in range(
        args.start_idx,
        args.end_idx + 1,
    ):

        pt_path = os.path.join(
            args.pt_dir,
            f"{idx:08d}.pt",
        )

        if not os.path.exists(pt_path):
            print(
                f"Skip missing file: {pt_path}"
            )
            continue

        (
            image_hidden_layers,
            last_token_all_layers,
        ) = load_hidden_data(
            pt_path,
            args.normalize,
        )

        if image_hidden_layers is None:
            print(
                f"Skip no-image file: {pt_path}"
            )
            continue

        # =====================================================
        # Per-sample image-token effective rank
        #
        # [L, N, D]
        #     ->
        # [L]
        # =====================================================
        sample_layer_eranks = (
            compute_per_sample_layer_eranks(
                image_hidden_layers,
                device,
            )
        )

        per_sample_layer_eranks.append(
            sample_layer_eranks
        )

        image_hidden_samples.append(
            image_hidden_layers
        )

        # =====================================================
        # [L, D]
        # =====================================================
        last_token_all_layers_samples.append(
            last_token_all_layers
        )

        loaded_files.append(
            pt_path
        )

    # =========================================================
    # Sanity check
    # =========================================================
    if not per_sample_layer_eranks:
        raise RuntimeError(
            "No image tokens loaded."
        )

    # =========================================================
    # Per-sample image-token eRank
    #
    # list of:
    # [L]
    #
    # ->
    #
    # [B, L]
    # =========================================================
    per_sample_layer_eranks = torch.stack(
        per_sample_layer_eranks,
        dim=0,
    )

    # =========================================================
    # Average eRank of image tokens within each sample
    #
    # [B, L]
    # ->
    # [L]
    # =========================================================
    avg_per_sample_layer_eranks = (
        per_sample_layer_eranks.mean(
            dim=0
        )
    )

    std_per_sample_layer_eranks = (
        per_sample_layer_eranks.std(
            dim=0,
            unbiased=False,
        )
    )

    # =========================================================
    # LAST TOKEN ACROSS ALL SAMPLES
    #
    # each sample:
    # [L, D]
    #
    # stack:
    # [B, L, D]
    # =========================================================
    last_token_all_layers_samples = torch.stack(
        last_token_all_layers_samples,
        dim=0,
    )

    num_samples = (
        last_token_all_layers_samples.size(0)
    )

    num_hidden_layers = (
        last_token_all_layers_samples.size(1)
    )

    hidden_dim = (
        last_token_all_layers_samples.size(2)
    )

    # =========================================================
    # Calculate last-token eRank for EVERY hidden layer
    #
    # layer 0:
    #   [B, D]
    #
    # layer 1:
    #   [B, D]
    #
    # ...
    #
    # Calculate:
    #   raw eRank
    #   normalized eRank
    # =========================================================
    last_token_layer_eranks_raw = []
    last_token_layer_eranks_norm = []

    for layer_idx in range(
        num_hidden_layers
    ):

        # =====================================================
        # Last token at current layer across all samples
        #
        # [B, L, D]
        #       |
        #       v
        # [B, D]
        # =====================================================
        layer_last_tokens = (
            last_token_all_layers_samples[
                :,
                layer_idx,
                :,
            ]
        )

        # =====================================================
        # 1. RAW
        # =====================================================
        raw_erank = compute_effective_rank(
            layer_last_tokens.to(device)
        ).cpu()

        last_token_layer_eranks_raw.append(
            raw_erank
        )

        # =====================================================
        # 2. NORMALIZED
        #
        # Normalize each sample vector independently.
        #
        # [B, D]
        # =====================================================
        layer_last_tokens_norm = F.normalize(
            layer_last_tokens,
            p=2,
            dim=-1,
        )

        norm_erank = compute_effective_rank(
            layer_last_tokens_norm.to(device)
        ).cpu()

        last_token_layer_eranks_norm.append(
            norm_erank
        )

    # [L]
    last_token_layer_eranks_raw = torch.stack(
        last_token_layer_eranks_raw,
        dim=0,
    )

    # [L]
    last_token_layer_eranks_norm = torch.stack(
        last_token_layer_eranks_norm,
        dim=0,
    )

    # =========================================================
    # Dataset-level effective ranks for IMAGE TOKENS
    # =========================================================
    all_token_layer_eranks = []
    mean_pooled_layer_eranks = []

    num_layers = (
        image_hidden_samples[0].size(0)
    )

    for layer_idx in range(
        num_layers
    ):

        # =====================================================
        # 1. All image tokens across all images
        #
        # image 1:
        # [N1, D]
        #
        # image 2:
        # [N2, D]
        #
        # ...
        #
        # concatenate:
        #
        # [N1 + N2 + ... + NB, D]
        # =====================================================
        all_image_tokens = torch.cat(
            [
                image_hidden[layer_idx]
                for image_hidden
                in image_hidden_samples
            ],
            dim=0,
        ).to(device)

        all_token_erank = (
            compute_effective_rank(
                all_image_tokens
            ).cpu()
        )

        all_token_layer_eranks.append(
            all_token_erank
        )

        # =====================================================
        # 2. Mean pool image tokens within EACH image
        #
        # image 1:
        # [N1, D] -> [D]
        #
        # image 2:
        # [N2, D] -> [D]
        #
        # ...
        #
        # stack:
        #
        # [B, D]
        #
        # then compute eRank across images
        # =====================================================
        mean_pooled_images = torch.stack(
            [
                image_hidden[
                    layer_idx
                ].mean(dim=0)
                for image_hidden
                in image_hidden_samples
            ],
            dim=0,
        ).to(device)

        mean_pooled_erank = (
            compute_effective_rank(
                mean_pooled_images
            ).cpu()
        )

        mean_pooled_layer_eranks.append(
            mean_pooled_erank
        )

    # [L]
    all_token_layer_eranks = torch.stack(
        all_token_layer_eranks,
        dim=0,
    )

    # [L]
    mean_pooled_layer_eranks = torch.stack(
        mean_pooled_layer_eranks,
        dim=0,
    )

    # =========================================================
    # Print basic information
    # =========================================================
    print(
        f"Loaded files: "
        f"{len(loaded_files)}"
    )

    print(
        "Per-sample image effective rank shape: "
        f"{tuple(per_sample_layer_eranks.shape)}"
    )

    print(
        "Last-token all-layers batch shape: "
        f"{tuple(last_token_all_layers_samples.shape)}"
    )

    print(
        f"Num samples: {num_samples}"
    )

    print(
        f"Num hidden layers: {num_hidden_layers}"
    )

    print(
        f"Hidden dim: {hidden_dim}"
    )

    # =========================================================
    # Print IMAGE TOKEN results
    # =========================================================
    print(
        "\n"
        "========================================\n"
        "IMAGE TOKEN EFFECTIVE RANK\n"
        "========================================"
    )

    print(
        "\nImage effective rank per layer:"
    )

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
            f"per_sample_mean="
            f"{mean_erank.item():.6f}, "
            f"per_sample_std="
            f"{std_erank.item():.6f}, "
            f"all_token_erank="
            f"{all_token_erank.item():.6f}, "
            f"mean_pooled_erank="
            f"{mean_pooled_erank.item():.6f}"
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
    # Print LAST TOKEN results
    # =========================================================
    print(
        "\n"
        "========================================\n"
        "LAST TOKEN EFFECTIVE RANK\n"
        "========================================"
    )

    print(
        "\nLast-token effective rank "
        "across samples per hidden layer:"
    )

    for (
        layer_idx,
        raw_erank,
        norm_erank,
    ) in zip(
        range(num_hidden_layers),
        last_token_layer_eranks_raw,
        last_token_layer_eranks_norm,
    ):

        print(
            f"  layer {layer_idx:02d}: "
            f"raw={raw_erank.item():.6f}, "
            f"normalized={norm_erank.item():.6f}"
        )

    # =========================================================
    # Last layer summary
    # =========================================================
    print(
        "\nLast hidden layer - last token:"
    )

    print(
        "  raw effective rank:        "
        f"{last_token_layer_eranks_raw[-1].item():.6f}"
    )

    print(
        "  normalized effective rank: "
        f"{last_token_layer_eranks_norm[-1].item():.6f}"
    )


if __name__ == "__main__":
    main()