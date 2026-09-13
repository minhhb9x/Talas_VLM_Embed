import argparse
import os

import torch


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


def get_image_token_slice(
    obj: dict,
    hidden_state: torch.Tensor,
) -> slice:
    """Locate the image-token block in a saved, padding-free sequence."""

    num_image_tokens = int(obj.get("num_image_tokens", 0))
    num_valid_tokens = int(
        obj.get("num_valid_tokens", hidden_state.size(1))
    )

    if hidden_state.size(1) != num_valid_tokens:
        raise ValueError(
            f"Saved hidden length ({hidden_state.size(1)}) does not match "
            f"num_valid_tokens ({num_valid_tokens})."
        )

    if bool(obj.get("last_image_token", False)):
        # With a terminal EOS, the image block ends immediately before it.
        image_end = num_valid_tokens - int(
            bool(obj.get("has_eos_id", False))
        )
        image_start = image_end - num_image_tokens

    else:
        image_start = 0
        image_end = num_image_tokens

    if image_start < 0 or image_end > num_valid_tokens:
        raise ValueError(
            f"Invalid image-token range [{image_start}, {image_end}) for "
            f"num_valid_tokens={num_valid_tokens} and "
            f"num_image_tokens={num_image_tokens}."
        )

    return slice(image_start, image_end)


def load_image_hidden_layers(
    pt_path: str,
    normalize: bool = False,
) -> torch.Tensor | None:

    obj = torch.load(
        pt_path,
        map_location="cpu",
    )

    num_image_tokens = int(
        obj.get("num_image_tokens", 0)
    )

    if num_image_tokens <= 0:
        return None

    # [num_layers, num_valid_tokens, hidden_dim]
    hidden_state = obj["hidden_state"]

    image_slice = get_image_token_slice(
        obj,
        hidden_state,
    )

    # [num_layers, num_image_tokens, hidden_dim]
    image_hidden_layers = hidden_state[
        :,
        image_slice,
        :
    ].float()

    if image_hidden_layers.size(1) != num_image_tokens:
        raise ValueError(
            f"Extracted {image_hidden_layers.size(1)} image tokens "
            f"from {pt_path}, expected {num_image_tokens}."
        )

    if normalize:
        image_hidden_layers = torch.nn.functional.normalize(
            image_hidden_layers,
            p=2,
            dim=-1,
        )

    return image_hidden_layers


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


def get_pt_files(
    pt_dir: str,
    num_samples: int,
) -> list[str]:
    """
    Get the first num_samples .pt files according to file modification time.

    This is useful when inference was shuffled, because filenames correspond
    to the original dataset indices rather than inference order.
    """

    if not os.path.isdir(pt_dir):
        raise FileNotFoundError(
            f"PT directory does not exist: {pt_dir}"
        )

    pt_files = [
        os.path.join(pt_dir, filename)
        for filename in os.listdir(pt_dir)
        if filename.endswith(".pt")
    ]

    if not pt_files:
        raise RuntimeError(
            f"No .pt files found in: {pt_dir}"
        )

    # Earlier-created / written files first.
    pt_files.sort()
    
    if num_samples > 0:
        pt_files = pt_files[:num_samples]

    return pt_files


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pt_dir",
        default=(
            "infer/"
            "FastVLM-0.5B_talas_1.0_eos_cls/"
            "ImageNet-1K/query"
        ),
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help=(
            "Number of first .pt files to use according to file "
            "modification time. Use <= 0 to process all files."
        ),
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="effective_rank_results.txt",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "L2-normalize each image token along the hidden dimension "
            "before computing effective rank."
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        args.device
    )

    per_sample_layer_eranks = []
    image_hidden_samples = []
    loaded_files = []

    # =========================================================
    # Get first N generated .pt files
    # =========================================================
    pt_files = get_pt_files(
        pt_dir=args.pt_dir,
        num_samples=args.num_samples,
    )

    print(
        f"Found {len(pt_files)} .pt files to process."
    )

    print(
        "First files:"
    )

    for pt_path in pt_files[:10]:
        print(
            f"  {os.path.basename(pt_path)}"
        )

    if len(pt_files) > 10:
        print("  ...")

    # =========================================================
    # Load samples
    # =========================================================
    for file_idx, pt_path in enumerate(
        pt_files
    ):
        print(
            f"[{file_idx + 1}/{len(pt_files)}] "
            f"Loading {os.path.basename(pt_path)}"
        )

        image_hidden_layers = load_image_hidden_layers(
            pt_path,
            normalize=args.normalize,
        )

        if image_hidden_layers is None:
            print(
                f"Skip no-image file: {pt_path}"
            )
            continue

        per_sample_layer_eranks.append(
            compute_per_sample_layer_eranks(
                image_hidden_layers,
                device,
            )
        )

        image_hidden_samples.append(
            image_hidden_layers
        )

        loaded_files.append(
            pt_path
        )

    if not per_sample_layer_eranks:
        raise RuntimeError(
            "No image tokens loaded."
        )

    # [num_samples, num_layers]
    per_sample_layer_eranks = torch.stack(
        per_sample_layer_eranks,
        dim=0,
    )

    # =========================================================
    # Average per-sample effective rank
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
    # Dataset-level effective ranks
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
                image_hidden[layer_idx].mean(
                    dim=0
                )
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

    all_token_layer_eranks = torch.stack(
        all_token_layer_eranks,
        dim=0,
    )

    mean_pooled_layer_eranks = torch.stack(
        mean_pooled_layer_eranks,
        dim=0,
    )

    # =========================================================
    # Print + Save results
    # =========================================================
    output_lines = []

    output_lines.append(
        f"Requested samples: {args.num_samples}"
    )

    output_lines.append(
        f"Loaded files: {len(loaded_files)}"
    )

    output_lines.append(
        f"L2-normalized image tokens: {args.normalize}"
    )

    output_lines.append(
        "File selection order: modification time (oldest first)"
    )

    output_lines.append(
        "Per-sample image effective rank shape: "
        f"{tuple(per_sample_layer_eranks.shape)}"
    )

    output_lines.append("")

    output_lines.append(
        "Loaded sample files:"
    )

    for pt_path in loaded_files:
        output_lines.append(
            f"  {os.path.basename(pt_path)}"
        )

    output_lines.append("")

    output_lines.append(
        "Image effective rank per layer:"
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

        output_lines.append(
            f"  layer {layer_idx:02d}: "
            f"per_sample_mean={mean_erank.item():.6f}, "
            f"per_sample_std={std_erank.item():.6f}, "
            f"all_token_erank={all_token_erank.item():.6f}, "
            f"mean_pooled_erank={mean_pooled_erank.item():.6f}"
        )

    output_lines.append("")

    output_lines.append(
        "Last layer per-sample mean effective rank: "
        f"{avg_per_sample_layer_eranks[-1].item():.6f}"
    )

    output_lines.append(
        "Last layer all-token effective rank: "
        f"{all_token_layer_eranks[-1].item():.6f}"
    )

    output_lines.append(
        "Last layer mean-pooled effective rank: "
        f"{mean_pooled_layer_eranks[-1].item():.6f}"
    )

    output_text = "\n".join(
        output_lines
    )

    # Print to terminal
    print()
    print(output_text)

    # Create output directory if needed
    output_dir = os.path.dirname(
        args.output_file
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    # Save to file
    with open(
        args.output_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            output_text + "\n"
        )

    print()
    print(
        f"Saved output to: {args.output_file}"
    )


if __name__ == "__main__":
    main()