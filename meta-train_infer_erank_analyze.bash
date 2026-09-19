#!/bin/bash

INFER_SCRIPT="infer_eval_hidden_attention.py"   # sửa thành script infer của bạn

INFER_SUBSETS=(
    "ImageNet-1K"
    # "N24News" "HatefulMemes" "VOC2007" "SUN397"
    # "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211"
)

MODELS=(
    # "meta_train/rkd_meta_cls/checkpoint-epoch-0"
    # "meta_train/ckd_meta_cls/checkpoint-epoch-0"
    # "meta_train/norm_meta_cls/checkpoint-epoch-0"
    "meta_train/rkd_meta_llavaov_cls/checkpoint-epoch-0"
    "meta_train/ckd_meta_llavaov_cls/checkpoint-epoch-0"
    "meta_train/norm_meta_llavaov_cls/checkpoint-epoch-0"
)

BACKBONES=(
    # "llava_qwen2"
    # "llava_qwen2"
    # "llava_qwen2"
    "llava_onevision"
    "llava_onevision"
    "llava_onevision"
)

# Kiểm tra MODELS và BACKBONES có cùng số phần tử không
if [ "${#MODELS[@]}" -ne "${#BACKBONES[@]}" ]; then
    echo "Error: MODELS and BACKBONES must have the same length."
    exit 1
fi

mkdir -p infer
mkdir -p analyze

for i in "${!MODELS[@]}"; do

    MODEL="${MODELS[$i]}"
    BACKBONE="${BACKBONES[$i]}"

    EXP_NAME=$(basename "$(dirname "$MODEL")")

    EXTRA_ARGS=()
    OUTPUT_SUFFIX=""

    if [ "$BACKBONE" = "llava_onevision" ]; then
        IMAGE_RESOLUTION="tiny"

        EXTRA_ARGS+=(--image_resolution "$IMAGE_RESOLUTION")
        OUTPUT_SUFFIX="_${IMAGE_RESOLUTION}"
    fi

    INFER_OUTPUT="infer/${EXP_NAME}${OUTPUT_SUFFIX}"
    ANALYZE_OUTPUT="analyze/${EXP_NAME}${OUTPUT_SUFFIX}.txt"

    echo "========================================================"
    echo "Model     : $MODEL"
    echo "Backbone  : $BACKBONE"
    echo "Experiment: $EXP_NAME"
    echo "Infer out : $INFER_OUTPUT"
    echo "Analyze   : $ANALYZE_OUTPUT"
    echo "Extra args: ${EXTRA_ARGS[*]}"
    echo "========================================================"

    CUDA_VISIBLE_DEVICES=0 python "$INFER_SCRIPT" \
        --model_name "$MODEL" \
        --lora True \
        --lora_r 64 \
        --lora_alpha 64 \
        --pooling eos \
        --model_backbone "$BACKBONE" \
        --normalize True \
        --bf16 \
        --dataset_name "TIGER-Lab/MMEB-eval" \
        --subset_name "${INFER_SUBSETS[@]}" \
        --dataset_split "test" \
        --image_dir "eval_images/" \
        --tgt_prefix_mod \
        --encode_output_path "$INFER_OUTPUT" \
        --per_device_eval_batch_size 1 \
        --load_pretrained_lora True \
        --report_to None \
        "${EXTRA_ARGS[@]}"

    if [ $? -ne 0 ]; then
        echo "ERROR: Infer failed for $EXP_NAME"
        continue
    fi

    CUDA_VISIBLE_DEVICES=0 python er_statistic.py \
        --pt_dir "${INFER_OUTPUT}/${INFER_SUBSETS[0]}/query" \
        --num_samples 3 \
        --normalize \
        --output_file "$ANALYZE_OUTPUT"

    echo "Finished: ${EXP_NAME}${OUTPUT_SUFFIX}"
    echo

done