#!/bin/bash
set -e

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --embed_map_model)
      embed_map_model="$2"
      shift
      shift
      ;;
    --watermark_model)
      watermark_model="$2"
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


rm -f watermarking/models_cl.py
cp contrastive_train/contrastive_model/models.py watermarking/models_cl.py


# ===== watermarking =====
python watermarking/generation_1step_end2end.py \
    --embed_map_model=$embed_map_model \
    --watermark_model=$watermark_model \
    --output_file=${watermark_output_file} \
    --alpha=${alpha} --delta_0=$delta_0 --delta=$delta \
    --data_path=${data_path} \
    --data_size=${data_size}
