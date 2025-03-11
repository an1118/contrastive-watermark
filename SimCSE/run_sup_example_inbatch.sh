#!/bin/bash

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --gpu_id)
      gpu_id="$2"
      shift
      shift
      ;;
    --output_dir)
      output_dir="$2"
      shift
      shift
      ;;
    --model_name)
      model_name="$2"
      shift
      shift
      ;;
    --num_paraphrased_llama)
      num_paraphrased_llama="$2"
      shift
      shift
      ;;
    --num_paraphrased_gpt)
      num_paraphrased_gpt="$2"
      shift
      shift
      ;;
    --num_sentiment_spoof)
      num_sentiment_spoof="$2"
      shift
      shift
      ;;
    --num_latter_sentiment_spoof)
      num_latter_sentiment_spoof="$2"
      shift
      shift
      ;;
    --num_factual_spoof)
      num_factual_spoof="$2"
      shift
      shift
      ;;
    --num_hate)
      num_hate="$2"
      shift
      shift
      ;;
    --train_file)
      train_file="$2"
      shift
      shift
      ;;
    --valid_file)
      valid_file="$2"
      shift
      shift
      ;;
    --dataset)
      dataset="$2"
      shift
      shift
      ;;
    --batch_size)
      batch_size="$2"
      shift
      shift
      ;;
    --train_epochs)
      train_epochs="$2"
      shift
      shift
      ;;
    --pooler_type)
      pooler_type="$2"
      shift
      shift
      ;;
    --freeze_base)
      freeze_base="$2"
      shift
      shift
      ;;
    --cl_weight)
      cl_weight="$2"
      shift
      shift
      ;;
    --tl_weight)
      tl_weight="$2"
      shift
      shift
      ;;
    --neg_weight)
      neg_weight="$2"
      shift
      shift
      ;;
    --margin)
      margin="$2"
      shift
      shift
      ;;
    *)
      shift
      ;;
  esac
done


model_name_="${model_name#*/}"
if [[ "$model_name_" == "gte-Qwen2-1.5B-instruct" ]]; then
    model_name_="${model_name_}-${pooler_type}"
fi

# Append '-freeze' to model_name_ if freeze_base is True
if [ "$freeze_base" == "True" ]; then
    model_name_="${model_name_}-freeze"
fi

if (( $(echo "$cl_weight != 0.0" | bc -l) )); then
  # HARD_NEGATIVE_WEIGHT=$(python3 -c "import math; print(math.log(${neg_weight}))")
  HARD_NEGATIVE_WEIGHT=$neg_weight
  echo "HARD_NEGATIVE_WEIGHT: $HARD_NEGATIVE_WEIGHT"
else
  HARD_NEGATIVE_WEIGHT=999
fi


ACCELERATE_LOG_LEVEL=info accelerate launch --config_file SimCSE/simcse_config.yaml SimCSE/train.py \
    --model_name_or_path ${model_name} \
    --train_file ${train_file} \
    --validation_file ${valid_file} \
    --output_dir ${output_dir} \
    --hard_negative_weight $HARD_NEGATIVE_WEIGHT \
    --cl_weight $cl_weight \
    --tl_weight $tl_weight \
    --margin $margin \
    --num_paraphrased_llama $num_paraphrased_llama \
    --num_paraphrased_gpt $num_paraphrased_gpt \
    --num_sentiment_spoof $num_sentiment_spoof \
    --num_latter_sentiment_spoof $num_latter_sentiment_spoof \
    --num_factual_spoof $num_factual_spoof \
    --num_hate $num_hate \
    --num_train_epochs $train_epochs \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --learning_rate 3e-5 \
    --max_seq_length 450 \
    --freeze_base $freeze_base \
    --evaluation_strategy steps \
    --save_strategy best \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model loss \
    --eval_steps 5 \
    --pooler_type ${pooler_type} \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --gradient_checkpointing \
    --report_to="wandb" \
    --run_name="${model_name_}-${dataset}-loss_cl${cl_weight}-tl${tl_weight}-wneg${neg_weight}-margin${margin}-llama${num_paraphrased_llama}gpt${num_paraphrased_gpt}-sent${num_sentiment_spoof}-latter_sent${num_latter_sentiment_spoof}-fact${num_factual_spoof}-hate${num_hate}" \
    --logging_steps=1 \
    "$@"
