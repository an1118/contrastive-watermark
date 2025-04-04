#!/bin/bash

module load cuda

set -e
repo="/blue/buyuheng/li_an.ucsb/projects/contrastive-watermark"

# model parameters
model_name="cardiffnlp/twitter-roberta-base-sentiment"

# training data parameters
dataset_name="annnli/C4-contrastive-watermark"
num_paraphrased_llama=8
num_paraphrased_gpt=8
num_sentiment_spoof=1
num_latter_sentiment_spoof=1
num_hate=1

# training parameters
batch_size=64
train_epochs=15
# loss function parameters
margin=0.9

# watermarking parameters
watermark_data_path="https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/realnewslike/c4-train.00000-of-00512.json.gz"
watermark_model="meta-llama/Llama-3.1-8B-Instruct"  # Qwen/Qwen2.5-7B-Instruct  meta-llama/Llama-3.1-8B-Instruct
data_size=200
alpha=2.0
delta_0=0.1
delta=0.13

model_name_=$(basename "$model_name")


echo "========Running following parameters combination...========"
echo "dataset_name: ${dataset_name}"
echo "Llama paraphrased: $num_paraphrased_llama; GPT paraphrased: $num_paraphrased_gpt"
echo "Sentiment spoofing: $num_sentiment_spoof; Latter sentiment spoofing: $num_latter_sentiment_spoof"
echo "Hate speech spoofing: $num_hate"
echo "model: $model_name_"
echo "epochs: $train_epochs; batch_size: $batch_size"
echo "margin: $margin"
echo "watermarking model: $watermark_model"
echo "watermarking parameters: alpha: $alpha, delta_0: $delta_0, delta: $delta"
echo "============================================================"

embed_map_model="${repo}/contrastive_train/result/${batch_size}batch_${train_epochs}epochs/llama${num_paraphrased_llama}gpt${num_paraphrased_gpt}-sent${num_sentiment_spoof}-latter_sent${num_latter_sentiment_spoof}-hate${num_hate}/loss_margin${margin}"
bash contrastive_train/run_contrastive_train.sh \
  --dataset_name $dataset_name \
  --num_paraphrased_llama $num_paraphrased_llama \
  --num_paraphrased_gpt $num_paraphrased_gpt \
  --num_sentiment_spoof $num_sentiment_spoof \
  --num_latter_sentiment_spoof $num_latter_sentiment_spoof \
  --num_hate $num_hate \
  --model_name $model_name \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --margin $margin \
  --output_dir $embed_map_model


echo "=========== watermarking on C4 ==========="

watermark_output_file="$repo/watermarking/outputs/wm-model-$(basename "$watermark_model")/wm-alpha${alpha}-delta${delta_0}|${delta}.csv"

bash watermarking/run_watermark.sh \
  --embed_map_model $embed_map_model \
  --watermark_model $watermark_model \
  --data_path $watermark_data_path --data_size $data_size \
  --watermark_output_file $watermark_output_file \
  --alpha $alpha --delta_0 $delta_0 --delta $delta
