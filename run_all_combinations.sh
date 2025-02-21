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
num_negative_llama=0
num_negative_gpt=1
num_summary=0

# training parameters
batch_size=128
train_epochs=30
# loss function parameters
cl_weight=0.0
tl_weight=1.0
neg_weight=1
margin=0.6

# watermarking parameters
watermark_data_path="https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/realnewslike/c4-train.00000-of-00512.json.gz"
data_size=200
alpha=2.0
delta_0=0.2
delta=0.5

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

echo "========Running following parameters combination...========"
echo "train_file: ${data_path_prefix}-train.csv"
echo "valid_file: ${data_path_prefix}-valid.csv"
echo "Llama paraphrased: $num_paraphrased_llama; GPT paraphrased: $num_paraphrased_gpt"
echo "Llama spoofing: $num_negative_llama; GPT spoofing: $num_negative_gpt"
echo "model: $model_name_"
echo "epochs: $train_epochs; batch_size: $batch_size"
echo "cl_weight: $cl_weight; tl_weight: $tl_weight"
echo "neg_weight: $neg_weight; margin: $margin"
echo "watermarking parameters: alpha: $alpha, delta_0: $delta_0, delta: $delta"
echo "============================================================"

embed_map_model="${repo}/SimCSE/result/${dataset}/${model_name_}/${batch_size}batch_${train_epochs}epochs/llama${num_paraphrased_llama}-${num_negative_llama}gpt${num_paraphrased_gpt}-${num_negative_gpt}-${num_summary}/loss_cl${cl_weight}-tl${tl_weight}-wneg${neg_weight}-margin${margin}"
# bash SimCSE/run_sup_example_inbatch.sh \
#   --gpu_id $gpu_id \
#   --train_file ${data_path_prefix}-train.csv \
#   --valid_file ${data_path_prefix}-valid.csv \
#   --dataset $dataset \
#   --num_paraphrased_llama $num_paraphrased_llama \
#   --num_paraphrased_gpt $num_paraphrased_gpt \
#   --num_negative_llama $num_negative_llama \
#   --num_negative_gpt $num_negative_gpt \
#   --num_summary $num_summary \
#   --model_name $model_name \
#   --pooler_type $pooler_type \
#   --freeze_base $freeze_base \
#   --batch_size $batch_size \
#   --train_epochs $train_epochs \
#   --cl_weight $cl_weight \
#   --tl_weight $tl_weight \
#   --neg_weight $neg_weight \
#   --margin $margin \
#   --output_dir $embed_map_model


echo "=========== watermarking on C4 ==========="
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

watermark_output_dir="$repo/watermarking/outputs/${dataset}/${model_name_}/${batch_size}batch_${train_epochs}epochs/llama${num_paraphrased_llama}-${num_negative_llama}gpt${num_paraphrased_gpt}-${num_negative_gpt}-${num_summary}/loss_cl${cl_weight}-tl${tl_weight}-wneg${neg_weight}-margin${margin}"

watermark_output_file="$watermark_output_dir/wm-${wm_dataset_name}-alpha${alpha}-delta${delta_0}|${delta}.csv"
eda_output_file="$watermark_output_dir/wm-${wm_dataset_name}-alpha${alpha}-delta${delta_0}|${delta}-sim.csv"

bash watermarking/run_watermark.sh \
  --gpu_id $gpu_id \
  --embed_map_model $embed_map_model \
  --data_path $watermark_data_path --data_size $data_size \
  --watermark_output_file $watermark_output_file \
  --eda_output_file $eda_output_file \
  --alpha $alpha --delta_0 $delta_0 --delta $delta


echo "=========== watermarking on LFQA ==========="
watermark_data_path="/blue/buyuheng/li_an.ucsb/projects/watermark-simcse/data/lfqa.json"

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

watermark_output_dir="$repo/watermarking/outputs/${dataset}/${model_name_}/${batch_size}batch_${train_epochs}epochs/llama${num_paraphrased_llama}-${num_negative_llama}gpt${num_paraphrased_gpt}-${num_negative_gpt}-${num_summary}/loss_cl${cl_weight}-tl${tl_weight}-wneg${neg_weight}-margin${margin}"

watermark_output_file="$watermark_output_dir/wm-${wm_dataset_name}-alpha${alpha}-delta${delta_0}|${delta}.csv"
eda_output_file="$watermark_output_dir/wm-${wm_dataset_name}-alpha${alpha}-delta${delta_0}|${delta}-sim.csv"

bash watermarking/run_watermark.sh \
  --gpu_id $gpu_id \
  --embed_map_model $embed_map_model \
  --data_path $watermark_data_path --data_size $data_size \
  --watermark_output_file $watermark_output_file \
  --eda_output_file $eda_output_file \
  --alpha $alpha --delta_0 $delta_0 --delta $delta


echo "=========== watermarking on IMDB ==========="
watermark_data_path="/blue/buyuheng/li_an.ucsb/projects/watermark-simcse/data/imdb-simcse-filtered-test.csv"

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

watermark_output_dir="$repo/watermarking/outputs/${dataset}/${model_name_}/${batch_size}batch_${train_epochs}epochs/llama${num_paraphrased_llama}-${num_negative_llama}gpt${num_paraphrased_gpt}-${num_negative_gpt}-${num_summary}/loss_cl${cl_weight}-tl${tl_weight}-wneg${neg_weight}-margin${margin}"

watermark_output_file="$watermark_output_dir/wm-${wm_dataset_name}-alpha${alpha}-delta${delta_0}|${delta}.csv"
eda_output_file="$watermark_output_dir/wm-${wm_dataset_name}-alpha${alpha}-delta${delta_0}|${delta}-sim.csv"

bash watermarking/run_watermark.sh \
  --gpu_id $gpu_id \
  --embed_map_model $embed_map_model \
  --data_path $watermark_data_path --data_size $data_size \
  --watermark_output_file $watermark_output_file \
  --eda_output_file $eda_output_file \
  --alpha $alpha --delta_0 $delta_0 --delta $delta
