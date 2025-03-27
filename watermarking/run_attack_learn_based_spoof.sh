#!/bin/bash
#SBATCH --job-name=watermark
#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
##SBATCH --reservation=buyuheng 
#SBATCH --gpus=a100:1
#SBATCH --mem=64gb
#SBATCH --time=20:00:00
#SBATCH --exclude=c0906a-s29

module load cuda

set -e
repo="/blue/buyuheng/li_an.ucsb/projects/watermark-simcse"
gpu_id=4

# model parameters
model_name="cardiffnlp/twitter-roberta-base-sentiment"  # Alibaba-NLP/gte-Qwen2-1.5B-instruct    cardiffnlp/twitter-roberta-base-sentiment
freeze_base=False
pooler_type="attention"

# training data parameters
dataset=c4
data_path_prefix="$repo/data/$dataset-simcse-filtered"
num_paraphrased_llama=8
num_paraphrased_gpt=8
num_sentiment_spoof=1
num_latter_sentiment_spoof=1
num_factual_spoof=0
num_hate=1

# training parameters
batch_size=64
train_epochs=15
# loss function parameters
cl_weight=0.0
tl_weight=1.0
neg_weight=1
margin=0.9

# watermarking parameters
watermark_data_path="https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/realnewslike/c4-train.00000-of-00512.json.gz"
watermark_model="meta-llama/Llama-3.1-8B-Instruct"  # Qwen/Qwen2.5-7B-Instruct  meta-llama/Llama-3.1-8B-Instruct
data_size=200
alpha=2.0
delta_0=0.1
delta=0.13
num_queries=1000

model_name_=$(basename "$model_name")
if [[ "$model_name_" == "gte-Qwen2-1.5B-instruct" ]]; then
    model_name_="${model_name_}-${pooler_type}"
fi
# Append '-freeze' to model_name_ if freeze_base is True
if [ "$freeze_base" == "True" ]; then
    model_name_="${model_name_}-freeze"
fi

if (( $(echo "$cl_weight != 0.0" | bc -l) )); then
  neg_weight=$neg_weight  # 1 32 64 128
else
  neg_weight=999
fi
if (( $(echo "$tl_weight != 0.0" | bc -l) )); then
  margin=$margin  # 0.2 0.5 0.8 1.1
else
  margin=999
fi

echo "========Run learn based spoofing attack========"
echo "========Running following parameters combination...========"
echo "train_file: ${data_path_prefix}-train.csv"
echo "valid_file: ${data_path_prefix}-valid.csv"
echo "Llama paraphrased: $num_paraphrased_llama; GPT paraphrased: $num_paraphrased_gpt"
echo "Sentiment spoofing: $num_sentiment_spoof; Latter sentiment spoofing: $num_latter_sentiment_spoof"
echo "Factual spoofing: $num_factual_spoof; Hate speech spoofing: $num_hate"
echo "model: $model_name_"
echo "epochs: $train_epochs; batch_size: $batch_size"
echo "cl_weight: $cl_weight; tl_weight: $tl_weight"
echo "neg_weight: $neg_weight; margin: $margin"
echo "watermarking model: $watermark_model"
echo "watermarking parameters: alpha: $alpha, delta_0: $delta_0, delta: $delta", num_queries: $num_queries
echo "============================================================"


rm -f watermarking/models_cl.py
cp SimCSE/simcse/models.py watermarking/models_cl.py

embed_map_model="${repo}/SimCSE/result/${dataset}/${model_name_}/${batch_size}batch_${train_epochs}epochs/llama${num_paraphrased_llama}gpt${num_paraphrased_gpt}-sent${num_sentiment_spoof}-latter_sent${num_latter_sentiment_spoof}-fact${num_factual_spoof}-hate${num_hate}/loss_cl${cl_weight}-tl${tl_weight}-wneg${neg_weight}-margin${margin}"

watermark_output_dir="$repo/watermarking/outputs/${dataset}/${model_name_}/${batch_size}batch_${train_epochs}epochs/llama${num_paraphrased_llama}gpt${num_paraphrased_gpt}-sent${num_sentiment_spoof}-latter_sent${num_latter_sentiment_spoof}-fact${num_factual_spoof}-hate${num_hate}/loss_cl${cl_weight}-tl${tl_weight}-wneg${neg_weight}-margin${margin}/wm-model-$(basename "$watermark_model")"



if [[ "${watermark_data_path,,}" == *"c4"* ]]; then
  wm_dataset_name="c4"
elif [[ "${watermark_data_path,,}" == *"imdb"* ]]; then
  wm_dataset_name="imdb"
elif [[ "${watermark_data_path,,}" == *"lfqa"* ]]; then
  wm_dataset_name="lfqa"
else
  echo "don't know how to handle dataset $watermark_data_path"
  exit 1
fi

watermark_output_file="$watermark_output_dir/attack_learn_based_spoof/${wm_dataset_name}-alpha${alpha}-delta${delta_0}|${delta}.csv"

# ===== watermarking =====
python watermarking/attack_learn_based_spoof.py \
    --embed_map_model=$embed_map_model \
    --watermark_model=$watermark_model \
    --output_file=${watermark_output_file} \
    --alpha=${alpha} --delta_0=$delta_0 --delta=$delta \
    --data_path=${watermark_data_path} \
    --num_queries=${num_queries} 
