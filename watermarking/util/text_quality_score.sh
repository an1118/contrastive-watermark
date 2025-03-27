#!/bin/bash
#SBATCH --job-name=quality
#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --time=20:00:00

module load cuda
set -e

repo="/blue/buyuheng/li_an.ucsb/projects/watermark-simcse/watermarking"

wm_dataset='c4'

# # kgw
# method="kgw"
# root="/blue/buyuheng/li_an.ucsb/projects/baselines/lm-watermarking/results/wm-model-Llama-3.1-8B-Instruct"
# generation_path="$root/$wm_dataset.csv"
# original_cname="original_text"
# wm_cname="watermarked_text"

# # unigram
# method="unigram"
# root="/blue/buyuheng/li_an.ucsb/projects/baselines/Unigram-Watermark/results/wm-model-Llama-3.1-8B-Instruct"
# generation_path="$root/${wm_dataset}_strength_2.0_frac_0.5_len_300.csv"
# original_cname="original_text"
# wm_cname="watermarked_text"

# # postmark
# method="postmark"
# root="/blue/buyuheng/li_an.ucsb/projects/baselines/PostMark/results"
# generation_path="$root/${wm_dataset}_0.06.csv"
# original_cname="text1"
# wm_cname="text2"


alpha=2.0
delta_0=0.1
delta=0.17

# adaptive
method="adaptive"
root="/blue/buyuheng/li_an.ucsb/projects/baselines/adaptive-text-watermark/outputs/wm-model-Llama-3.1-8B-Instruct"
generation_path="$root/wm-${wm_dataset}-alpha$alpha-delta$delta_0|$delta.csv"
original_cname="original_text"
wm_cname="adaptive_watermarked_text"

# # ours
# method="ours"
# root="/blue/buyuheng/li_an.ucsb/projects/watermark-simcse/watermarking/outputs/c4/twitter-roberta-base-sentiment/64batch_15epochs/llama8gpt8-sent1-latter_sent1-fact0-hate1/loss_cl0.0-tl1.0-wneg999-margin0.9/wm-model-Llama-3.1-8B-Instruct"
# generation_path="$root/wm-${wm_dataset}-alpha$alpha-delta$delta_0|$delta.csv"
# original_cname='original_text'
# wm_cname='adaptive_watermarked_text'


echo "=====parameters====="
echo "method: $method"
echo "wm_dataset: $wm_dataset"
if [[ "$method" == "adaptive" || "$method" == "ours" ]]; then
    echo "alpha: $alpha, delta_0: $delta_0, delta: $delta"
fi
echo "===================="


output_path="${generation_path%.csv}_text_quality.csv"
echo $output_path

python $repo/util/text_quality_score.py \
    --generation_path $generation_path \
    --original_cname $original_cname \
    --wm_cname $wm_cname \
    --output_path $output_path