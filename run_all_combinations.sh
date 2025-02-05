#!/bin/bash
set -e
repo="/mnt/data2/lian/projects/watermark/watermark-simcse"

model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct"  # "cardiffnlp/twitter-roberta-base-sentiment"
pooler_type="attention"
freeze_base=True
gpu_id=3
dataset=c4-imdb
batch_size=64
train_epochs=10
LOSS_FUNCTION_IDS=(2)  # 2 3 4
NEG_WEIGHTS=(1)  # 1 32 64 128
num_paraphrased_llama=4
num_paraphrased_gpt=4
num_negative_llama=1
num_negative_gpt=1
data_generation_model="mix_GPT-4o_Llama-3.1-8B-Instruct"
data_path_prefix="/mnt/data2/lian/projects/watermark/data/${data_generation_model}/${dataset}-simcse"
watermark_data_path="https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/realnewslike/c4-train.00000-of-00512.json.gz"

model_name_=$(basename "$model_name")

if [[ "$model_name_" == "gte-Qwen2-1.5B-instruct" ]]; then
    model_name_="${model_name_}-${pooler_type}"
fi

# Append '-freeze' to model_name_ if freeze_base is True
if [ "$freeze_base" == "True" ]; then
    model_name_="${model_name_}-freeze"
fi


for loss_function_id in "${LOSS_FUNCTION_IDS[@]}"; do
  for neg_weight in "${NEG_WEIGHTS[@]}"; do
    embed_map_model="${repo}/SimCSE/result/${model_name_}/${batch_size}batch_${train_epochs}epochs/llama${num_paraphrased_llama}-${num_negative_llama}gpt${num_paraphrased_gpt}-${num_negative_gpt}/end2end-${dataset}-loss_cl${loss_function_id}-wneg${neg_weight}"
    bash SimCSE/run_sup_example_inbatch.sh \
      --gpu_id $gpu_id \
      --output_dir $embed_map_model \
      --batch_size $batch_size \
      --train_epochs $train_epochs \
      --loss_function_id $loss_function_id \
      --num_paraphrased_llama $num_paraphrased_llama \
      --num_paraphrased_gpt $num_paraphrased_gpt \
      --num_negative_llama $num_negative_llama \
      --num_negative_gpt $num_negative_gpt \
      --neg_weight $neg_weight \
      --model_name $model_name \
      --train_file ${data_path_prefix}-train.csv \
      --valid_file ${data_path_prefix}-valid.csv \
      --pooler_type $pooler_type \
      --freeze_base $freeze_base
    
    watermark_output_dir="$repo/watermarking/outputs/end2end/$dataset/${model_name_}/${batch_size}batch_${train_epochs}epochs/llama${num_paraphrased_llama}-${num_negative_llama}gpt${num_paraphrased_gpt}-${num_negative_gpt}"

    bash watermarking/run_watermark.sh \
      --gpu_id $gpu_id \
      --watermark_output_dir $watermark_output_dir \
      --embed_map_model $embed_map_model \
      --neg_weight $neg_weight \
      --loss_function_id $loss_function_id \
      --data_path $watermark_data_path

  done
done


# # add watermarked text
# data_path="/mnt/data2/lian/projects/watermark/watermark-simcse/watermarking/outputs/end2end/c4/twitter-roberta-base-sentiment/128batch_2000epochs/sanity-check/llama4-1gpt4-1-wm/onebatch-c4-train-simcse-all-filtered-formatted.csv"

# embed_map_model="${repo}/SimCSE/result/${model_name_}/${batch_size}batch_${train_epochs}epochs/sanity-check/llama${num_paraphrased_llama}-${num_negative_llama}gpt${num_paraphrased_gpt}-${num_negative_gpt}-wm/end2end-c4-loss_cl${loss_function_id}-wneg${neg_weight}"


# for loss_function_id in "${LOSS_FUNCTION_IDS[@]}"; do
#   for neg_weight in "${NEG_WEIGHTS[@]}"; do
#     bash SimCSE/run_sup_example_inbatch.sh \
#       --gpu_id $gpu_id \
#       --output_dir $embed_map_model \
#       --batch_size $batch_size \
#       --train_epochs $train_epochs \
#       --loss_function_id $loss_function_id \
#       --num_paraphrased_llama $num_paraphrased_llama \
#       --num_paraphrased_gpt $num_paraphrased_gpt \
#       --num_negative_llama $num_negative_llama \
#       --num_negative_gpt $num_negative_gpt \
#       --neg_weight $neg_weight \
#       --model_name $model_name \
#       --train_file $data_path \
#       --pooler_type $pooler_type
    
#     watermark_output_dir="$repo/watermarking/outputs/end2end/$dataset/${model_name_}/${batch_size}batch_${train_epochs}epochs/sanity-check/llama${num_paraphrased_llama}-${num_negative_llama}gpt${num_paraphrased_gpt}-${num_negative_gpt}-wm"

#     bash watermarking/run_watermark.sh \
#       --gpu_id $gpu_id \
#       --watermark_output_dir $watermark_output_dir \
#       --embed_map_model $embed_map_model \
#       --neg_weight $neg_weight \
#       --loss_function_id $loss_function_id \
#       --data_path $data_path

#   done
# done
