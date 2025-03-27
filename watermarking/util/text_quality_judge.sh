

repo="/blue/buyuheng/li_an.ucsb/projects/watermark-simcse/watermarking"

wm_dataset='c4'
alpha=2.0
delta_0=0.1
delta=0.17
echo "=====parameters====="
echo "wm_dataset: $wm_dataset"
echo "alpha: $alpha, delta_0: $delta_0, delta: $delta"
echo "===================="

# # kgw
# baseline_name="kgw"
# baseline_result="/blue/buyuheng/li_an.ucsb/projects/baselines/lm-watermarking/results/wm-model-Llama-3.1-8B-Instruct/c4.csv"
# baseline_original_cname="original_text"
# baseline_wm_cname="watermarked_text"
# # unigram
# baseline_name="unigram"
# baseline_result="/blue/buyuheng/li_an.ucsb/projects/baselines/Unigram-Watermark/results/wm-model-Llama-3.1-8B-Instruct/c4_strength_2.0_frac_0.5_len_300.csv"
# baseline_original_cname="original_text"
# baseline_wm_cname="watermarked_text"
# adaptive
baseline_name="adaptive"
baseline_result="/blue/buyuheng/li_an.ucsb/projects/baselines/adaptive-text-watermark/outputs/wm-model-Llama-3.1-8B-Instruct/wm-c4-alpha$alpha-delta$delta_0|$delta.csv"
baseline_original_cname="original_text"
baseline_wm_cname="adaptive_watermarked_text"
# postmark
# baseline_name="postmark"
# baseline_result="/blue/buyuheng/li_an.ucsb/projects/baselines/PostMark/results/c4_0.06.csv"
# baseline_original_cname="text1"
# baseline_wm_cname="text2"

our_result="/blue/buyuheng/li_an.ucsb/projects/watermark-simcse/watermarking/outputs/c4/twitter-roberta-base-sentiment/64batch_15epochs/llama8gpt8-sent1-latter_sent1-fact0-hate1/loss_cl0.0-tl1.0-wneg999-margin0.9/wm-model-Llama-3.1-8B-Instruct/wm-${wm_dataset}-alpha$alpha-delta$delta_0|$delta.csv"
ours_original_cname='original_text'
ours_wm_cname='adaptive_watermarked_text'

echo "===== ${baseline_name} vs. ours ====="
output_path=$repo/outputs/text_quality_judge/delta$delta/${wm_dataset}_${baseline_name}_vs_ours.csv
mkdir -p $repo/outputs/text_quality_judge/delta$delta
python $repo/util/text_quality_judge.py \
    --baseline_path $baseline_result \
    --baseline_original_cname $baseline_original_cname \
    --baseline_cname $baseline_wm_cname \
    --ours_path $our_result \
    --ours_original_cname $ours_original_cname \
    --ours_cname $ours_wm_cname \
    --output_path $output_path