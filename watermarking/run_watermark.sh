#!/bin/bash
set -e

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --gpu_id)
      gpu_id="$2"
      shift
      shift
      ;;
    --embed_map_model)
      embed_map_model="$2"
      shift
      shift
      ;;
    --data_path)
      data_path="$2"
      shift
      shift
      ;;
    --data_size)
      data_size="$2"
      shift
      shift
      ;;
    --watermark_output_file)
      watermark_output_file="$2"
      shift
      shift
      ;;
    --eda_output_file)
      eda_output_file="$2"
      shift
      shift
      ;;
    --alpha)
      alpha="$2"
      shift
      shift
      ;;
    --delta_0)
      delta_0="$2"
      shift
      shift
      ;;
    --delta)
      delta="$2"
      shift
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# echo "Arguments received: $@"

rm -f watermarking/models_cl.py
cp SimCSE/simcse/models.py watermarking/models_cl.py

# eda_output_file="$watermark_output_dir/multiple-spoofing-attack.csv"  # for c4

# watermarking
# python watermarking/generation_1step_end2end.py \
#     --embed_map_model=$embed_map_model \
#     --output_file=${watermark_output_file} \
#     --alpha=${alpha} --delta_0=$delta_0 --delta=$delta \
#     --data_path=${data_path} \
#     --data_size=${data_size}
#     # --correct_grammar=false

# # ===== get similarity after 0-1 mapping =====
# echo "get similarity after 0-1 mapping.."
# python watermarking/eda_1step_end2end.py \
#     --embed_map_model=$embed_map_model \
#     --output_file=${eda_output_file} \
#     --alpha=${alpha} --delta_0=$delta_0 --delta=$delta \
#     --result_file=${watermark_output_file}

# ===== latter sentiment spoofing attack =====
echo "apply latter sentiment spoofing attack.."
python watermarking/generation_1step_end2end_latter_spoof.py \
    --embed_map_model=$embed_map_model \
    --output_file=${watermark_output_file} \
    --alpha=${alpha} --delta_0=$delta_0 --delta=$delta \
    --result_file=${watermark_output_file}

# # ===== multiple time attack =====
# CUDA_VISIBLE_DEVICES=$gpu_id python watermarking/eda_1step_end2end_multi-attack.py \
#     --embed_map_model=$embed_map_model \
#     --hard_negative_weight=$HARD_NEGATIVE_WEIGHT \
#     --output_file=${eda_output_file} \
#     --alpha=${alpha} --delta_0=$delta_0 --delta=$delta \
#     --result_file=${watermark_output_file}
