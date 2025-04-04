#!/bin/bash

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
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
    --num_hate)
      num_hate="$2"
      shift
      shift
      ;;
    --dataset_name)
      dataset_name="$2"
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


ACCELERATE_LOG_LEVEL=info accelerate launch --config_file contrastive_train/config.yaml contrastive_train/train.py \
    --model_name_or_path ${model_name} \
    --dataset_name ${dataset_name} \
    --output_dir ${output_dir} \
    --margin $margin \
    --num_paraphrased_llama $num_paraphrased_llama \
    --num_paraphrased_gpt $num_paraphrased_gpt \
    --num_sentiment_spoof $num_sentiment_spoof \
    --num_latter_sentiment_spoof $num_latter_sentiment_spoof \
    --num_hate $num_hate \
    --num_train_epochs $train_epochs \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --learning_rate 3e-5 \
    --max_seq_length 450 \
    --evaluation_strategy steps \
    --save_strategy best \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model loss \
    --eval_steps 5 \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --gradient_checkpointing \
    --report_to="wandb" \
    --run_name="${model_name_}-margin${margin}-llama${num_paraphrased_llama}gpt${num_paraphrased_gpt}-sent${num_sentiment_spoof}-latter_sent${num_latter_sentiment_spoof}-hate${num_hate}" \
    --logging_steps=1 \
    "$@"
