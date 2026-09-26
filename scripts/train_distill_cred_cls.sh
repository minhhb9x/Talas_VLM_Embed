#!/bin/bash


NUM_GPUS_PER_NODE=1


TRAIN_SCRIPT="train_ddp.py"

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
torchrun --standalone \
    --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name apple/FastVLM-0.5B \
    --lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data 1.0 \
    --output_dir "training/FastVLM-0.5B_cred_cls" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --lr_scheduler_type "constant" \
    --warmup_ratio 0.05 \
    --kd_weight 1.0 \
    --caching_dir "caching/B3_Qwen2_2B_cls" \
    --kd_loss_type "cred" \
    --image_resolution "low" \
    --projector_config_path "./config/projector_config_emo.json" \
    --projector_lr 5e-5 \
    --report_to None

echo "===================="
SUBSETS=(
  "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" 
  "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211"
  # "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
#   "ScienceQA" "VizWiz" "GQA" "TextVQA"
)
python eval_mmeb_2.py \
    --model_name training/FastVLM-0.5B_cred_cls/checkpoint-final \
    --encode_output_path './MMEB-eval_outputs/FastVLM-0.5B_cred_cls/' \
    --lora True --lora_r 64 --lora_alpha 64 \
    --pooling eos \
    --model_backbone "llava_qwen2" \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 16 \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --load_pretrained_lora True \
    --report_to none