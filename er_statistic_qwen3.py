import argparse
import os

import torch
import torch.nn.functional as F

from er_statistic import (
    compute_dataset_layer_eranks,
    compute_effective_rank,
    compute_per_sample_layer_eranks,
    get_pt_files,
    summarize_per_sample_eranks,
)


def _validate_token_mask(mask, name, num_tokens):
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    mask = mask.detach().cpu().bool().flatten()
    if mask.numel() != num_tokens:
        raise ValueError(
            f"{name} has {mask.numel()} positions, expected {num_tokens}."
        )
    return mask


def load_hidden_layers_qwen3(
    pt_path: str,
    normalize: bool = False,
):
    """
    Extract Qwen3 image/text hidden states using saved token-position masks.

    Image tokens may occupy one or multiple non-contiguous blocks. Text is
    selected by text_token_mask, which is the complement of image_token_mask
    over valid tokens. Therefore special tokens remain in text, and text spans
    before, between, and after image blocks are concatenated in original order.
    """
    obj = torch.load(pt_path, map_location="cpu")
    hidden_state = obj["hidden_state"].float()

    if hidden_state.ndim != 3:
        raise ValueError(
            f"hidden_state in {pt_path} must have shape [layers, tokens, dim], "
            f"got {tuple(hidden_state.shape)}."
        )

    num_tokens = hidden_state.size(1)
    image_mask = _validate_token_mask(
        obj.get("image_token_mask"), "image_token_mask", num_tokens
    )
    text_mask = _validate_token_mask(
        obj.get("text_token_mask"), "text_token_mask", num_tokens
    )

    if torch.any(image_mask & text_mask):
        raise ValueError(f"Image and text masks overlap in {pt_path}.")
    if not torch.all(image_mask | text_mask):
        raise ValueError(f"Image and text masks do not cover all tokens in {pt_path}.")

    num_image_tokens = int(image_mask.sum().item())
    if num_image_tokens <= 0:
        return None, None, None, None

    expected_image_tokens = int(obj.get("num_image_tokens", num_image_tokens))
    expected_text_tokens = int(obj.get("num_text_tokens", text_mask.sum().item()))
    if num_image_tokens != expected_image_tokens:
        raise ValueError(
            f"Image-mask count is {num_image_tokens}, but saved "
            f"num_image_tokens is {expected_image_tokens} in {pt_path}."
        )
    if int(text_mask.sum().item()) != expected_text_tokens:
        raise ValueError(
            f"Text-mask count is {int(text_mask.sum().item())}, but saved "
            f"num_text_tokens is {expected_text_tokens} in {pt_path}."
        )

    image_hidden_layers = hidden_state[:, image_mask, :]
    text_hidden_layers = hidden_state[:, text_mask, :]

    if text_hidden_layers.size(1) <= 0:
        raise ValueError(f"No text tokens were extracted from {pt_path}.")

    if normalize:
        image_hidden_layers = F.normalize(image_hidden_layers, p=2, dim=-1)
        text_hidden_layers = F.normalize(text_hidden_layers, p=2, dim=-1)

    last_token_all_layers = hidden_state[:, -1, :].clone()
    metadata = {
        "num_valid_tokens": num_tokens,
        "num_image_tokens": num_image_tokens,
        "num_text_tokens": int(text_mask.sum().item()),
        "num_special_text_tokens": int(obj.get("num_special_text_tokens", 0)),
        "image_positions": image_mask.nonzero(as_tuple=False).squeeze(1),
        "text_positions": text_mask.nonzero(as_tuple=False).squeeze(1),
    }
    return (
        image_hidden_layers,
        text_hidden_layers,
        last_token_all_layers,
        metadata,
    )


def _format_modality_section(
    title,
    per_sample_mean,
    per_sample_std,
    all_token_eranks,
    mean_pooled_eranks,
):
    lines = ["", f"{title} TOKEN EFFECTIVE RANK", f"{title.title()} effective rank per layer:"]
    for layer_idx, mean_erank, std_erank, all_token_erank, mean_pooled_erank in zip(
        range(per_sample_mean.numel()),
        per_sample_mean,
        per_sample_std,
        all_token_eranks,
        mean_pooled_eranks,
    ):
        lines.append(
            f"  layer {layer_idx:02d}: "
            f"per_sample_mean={mean_erank.item():.6f}, "
            f"per_sample_std={std_erank.item():.6f}, "
            f"all_token_erank={all_token_erank.item():.6f}, "
            f"mean_pooled_erank={mean_pooled_erank.item():.6f}"
        )

    name = title.lower()
    lines.extend(
        [
            "",
            f"Last layer {name} per-sample mean effective rank: "
            f"{per_sample_mean[-1].item():.6f}",
            f"Last layer {name} all-token effective rank: "
            f"{all_token_eranks[-1].item():.6f}",
            f"Last layer {name} mean-pooled effective rank: "
            f"{mean_pooled_eranks[-1].item():.6f}",
        ]
    )
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt_dir",
        default="infer/qwen3_vl/ImageNet-1K/query",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of first .pt files to use. Use <= 0 to process all files.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_file",
        type=str,
        default="effective_rank_results_qwen3.txt",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize image/text tokens before ERank.",
    )
    parser.add_argument(
        "--normalize_by_min_dim",
        action="store_true",
        help="Divide effective rank by min(n, d).",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    image_per_sample_eranks = []
    text_per_sample_eranks = []
    image_hidden_samples = []
    text_hidden_samples = []
    last_token_samples = []
    sample_metadata = []
    loaded_files = []

    pt_files = get_pt_files(args.pt_dir, args.num_samples)
    print(f"Found {len(pt_files)} .pt files to process.")
    print(
        "Token split: image_token_mask vs. all remaining valid tokens "
        "(special tokens retained)."
    )
    print(
        "Effective-rank normalization: "
        f"{'min(n, d)' if args.normalize_by_min_dim else 'none'}"
    )

    for file_idx, pt_path in enumerate(pt_files):
        print(f"[{file_idx + 1}/{len(pt_files)}] Loading {os.path.basename(pt_path)}")
        (
            image_hidden,
            text_hidden,
            last_tokens,
            metadata,
        ) = load_hidden_layers_qwen3(pt_path, normalize=args.normalize)

        if image_hidden is None:
            print(f"Skip no-image file: {pt_path}")
            continue

        image_per_sample_eranks.append(
            compute_per_sample_layer_eranks(
                image_hidden,
                device,
                normalize_by_min_dim=args.normalize_by_min_dim,
            )
        )
        text_per_sample_eranks.append(
            compute_per_sample_layer_eranks(
                text_hidden,
                device,
                normalize_by_min_dim=args.normalize_by_min_dim,
            )
        )
        image_hidden_samples.append(image_hidden)
        text_hidden_samples.append(text_hidden)
        last_token_samples.append(last_tokens)
        sample_metadata.append(metadata)
        loaded_files.append(pt_path)

    if not loaded_files:
        raise RuntimeError("No Qwen3 image/text token samples were loaded.")

    (
        image_per_sample_eranks,
        image_per_sample_mean,
        image_per_sample_std,
    ) = summarize_per_sample_eranks(image_per_sample_eranks)
    (
        text_per_sample_eranks,
        text_per_sample_mean,
        text_per_sample_std,
    ) = summarize_per_sample_eranks(text_per_sample_eranks)

    image_all_token_eranks, image_mean_pooled_eranks = (
        compute_dataset_layer_eranks(
            image_hidden_samples,
            device,
            normalize_by_min_dim=args.normalize_by_min_dim,
        )
    )
    text_all_token_eranks, text_mean_pooled_eranks = (
        compute_dataset_layer_eranks(
            text_hidden_samples,
            device,
            normalize_by_min_dim=args.normalize_by_min_dim,
        )
    )

    last_token_samples = torch.stack(last_token_samples, dim=0)
    last_token_eranks_raw = []
    last_token_eranks_normalized = []
    for layer_idx in range(last_token_samples.size(1)):
        layer_tokens = last_token_samples[:, layer_idx, :]
        last_token_eranks_raw.append(
            compute_effective_rank(
                layer_tokens.to(device),
                normalize_by_min_dim=args.normalize_by_min_dim,
            ).cpu()
        )
        last_token_eranks_normalized.append(
            compute_effective_rank(
                F.normalize(layer_tokens, p=2, dim=-1).to(device),
                normalize_by_min_dim=args.normalize_by_min_dim,
            ).cpu()
        )

    last_token_eranks_raw = torch.stack(last_token_eranks_raw)
    last_token_eranks_normalized = torch.stack(last_token_eranks_normalized)

    total_image_tokens = sum(item["num_image_tokens"] for item in sample_metadata)
    total_text_tokens = sum(item["num_text_tokens"] for item in sample_metadata)
    total_special_text_tokens = sum(
        item["num_special_text_tokens"] for item in sample_metadata
    )

    output_lines = [
        f"Requested samples: {args.num_samples}",
        f"Loaded files: {len(loaded_files)}",
        f"L2-normalized image/text tokens: {args.normalize}",
        "Image selection: exact Qwen3 <|image_pad|> positions",
        "Text selection: every valid non-image position",
        "Text special tokens retained: True",
        "Text spans around/between image blocks concatenated: True",
        f"Total image tokens: {total_image_tokens}",
        f"Total text tokens (including special tokens): {total_text_tokens}",
        f"Total special tokens included in text: {total_special_text_tokens}",
        "Effective-rank normalization: "
        f"{'min(n, d)' if args.normalize_by_min_dim else 'none'}",
        f"Per-sample image effective rank shape: {tuple(image_per_sample_eranks.shape)}",
        f"Per-sample text effective rank shape: {tuple(text_per_sample_eranks.shape)}",
        f"Last-token hidden-state shape: {tuple(last_token_samples.shape)}",
        "",
        "Loaded sample files:",
    ]
    output_lines.extend(f"  {os.path.basename(path)}" for path in loaded_files)

    output_lines.extend(
        _format_modality_section(
            "IMAGE",
            image_per_sample_mean,
            image_per_sample_std,
            image_all_token_eranks,
            image_mean_pooled_eranks,
        )
    )
    output_lines.extend(
        _format_modality_section(
            "TEXT",
            text_per_sample_mean,
            text_per_sample_std,
            text_all_token_eranks,
            text_mean_pooled_eranks,
        )
    )

    output_lines.extend(
        [
            "",
            "LAST TOKEN EFFECTIVE RANK",
            "Last-token effective rank across samples per hidden layer:",
        ]
    )
    for layer_idx, raw_erank, normalized_erank in zip(
        range(last_token_eranks_raw.numel()),
        last_token_eranks_raw,
        last_token_eranks_normalized,
    ):
        output_lines.append(
            f"  layer {layer_idx:02d}: "
            f"raw={raw_erank.item():.6f}, "
            f"normalized={normalized_erank.item():.6f}"
        )

    output_lines.extend(
        [
            "",
            "Last hidden layer last-token raw effective rank: "
            f"{last_token_eranks_raw[-1].item():.6f}",
            "Last hidden layer last-token normalized effective rank: "
            f"{last_token_eranks_normalized[-1].item():.6f}",
        ]
    )

    output_text = "\n".join(output_lines)
    print()
    print(output_text)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as output_file:
        output_file.write(output_text + "\n")
    print()
    print(f"Saved output to: {args.output_file}")


if __name__ == "__main__":
    main()
