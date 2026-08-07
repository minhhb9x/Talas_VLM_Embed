SUBSETS=(
  "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397"
    # "ImageNet-1K"
#   "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
)

# MODEL=training/FastVLM-0.5B_cls_0.3_talas/checkpoint-final
MODEL=training/FastVLM-0.5B_talas_1.0_eos_norm_proj_cls/checkpoint-epoch-0
export CUDA_VISIBLE_DEVICES=1
python eval_mmeb.py \
    --model_name $MODEL \
    --encode_output_path ./MMEB-eval_outputs/FastVLM-0.5B_talas_1.0_eos_norm_proj_cls/ \
    --lora True --lora_r 64 --lora_alpha 64 \
    --pooling eos \
    --model_backbone llava_qwen2 \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 32 \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --load_pretrained_lora True \
    --report_to none