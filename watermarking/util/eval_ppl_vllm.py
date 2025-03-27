import torch
from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm
from statistics import mean, median
from math import exp
import os
import re
import glob

import pdb


# Define perplexity calculation function
def calculate_text_perplexity(output):
    prompt_ids = output.prompt_token_ids
    null = []
    for idx, i in enumerate(output.prompt_logprobs):  # i is dict, {token_id: Logprob}
        if i is not None:
            token_id = prompt_ids[idx]
            null.append(i[token_id].logprob)
    perplexity = exp(-mean(null))
    return perplexity

def _get_perplexity(df, column):
    ppl_col = f'ppl_{column}'
    ppl_results = []
    outputs = model.generate(df[column], sampling_params)
    for output in tqdm(outputs, desc="Calculating Perplexity"):
        ppl_results.append(calculate_text_perplexity(output))
    df[ppl_col] = ppl_results
    return df[ppl_col]

def get_perplexity(generation_result_path, col_to_eval):
    df = pd.read_csv(generation_result_path)  # Load your DataFrame if needed
    print(f'Num of rows: {df.shape[0]}', flush=True)
    # col_to_eval = ['adaptive_watermarked_text']
    avg_ppl_dict = {}
    for col in col_to_eval:
        num_na = df[col].isna().sum()
        if num_na > 0:
            print(f'Replacing {num_na} empty values in column {col} with a space', flush=True)
            df[col].fillna(' ', inplace=True)
        print(f'===Column: {col}===', flush=True)
        # Calculate perplexity for each text in the DataFrame
        df[f'ppl_{col}'] = _get_perplexity(df, col)
        avg_ppl = round(df[f'ppl_{col}'].median(), 2)
        print(f'==> Median ppl of {col}:', avg_ppl, flush=True)
        avg_ppl_dict[f'ppl_{col}'] = avg_ppl
        df.to_csv(generation_result_path, index=False)
    return avg_ppl_dict


save_output = False
model_path = 'meta-llama/Llama-3.1-70B'
tp_size = 4
# model_path = 'meta-llama/Llama-3.2-3B-Instruct'
# tp_size = 1
gpu_memory_utilization=0.9
model = LLM(model=model_path, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_memory_utilization, max_model_len=1000, max_num_seqs=2)
sampling_params = SamplingParams(prompt_logprobs=1)

root_path = "/blue/buyuheng/li_an.ucsb/projects/watermark-simcse/watermarking/outputs/"

train_dataset_names = ['c4']
wm_dataset_names = ['c4']  # , 'lfqa'
wm_models = ['Llama-3.1-8B-Instruct']  # 'Qwen2.5-7B-Instruct'

results = []
for train_dataset_name in train_dataset_names:
    print(f'==================== train on {train_dataset_name} ====================', flush=True)
    for wm_model_name in wm_models:
        print(f'============ wm model {wm_model_name} ============', flush=True)
        for wm_dataset in wm_dataset_names:
            print(f'============ wm on {wm_dataset} ============', flush=True)
            # ## adaptive
            # print('--------adaptive-------', flush=True)
            # delta_values = [(0.1, 0.11), (0.1, 0.13), (0.1, 0.15), (0.2, 0.25), (0.2, 0.5)]
            # for delta_0, delta in delta_values:
            #     filepath = rf"/blue/buyuheng/li_an.ucsb/projects/baselines/adaptive-text-watermark/outputs/wm-model-{wm_model_name}/wm-{wm_dataset}-alpha2.0-delta{delta_0}|{delta}.csv"
            #     result_dict = {'wm_dataset_name': wm_dataset, 'wm_model': wm_model_name, 'method': 'adaptive', 'delta_0': delta_0, 'delta': delta}
            #     print(result_dict, flush=True)
            #     avg_ppl_dict = get_perplexity(filepath, ['adaptive_watermarked_text'])
            #     result_dict.update(avg_ppl_dict)
            #     results.append(result_dict)
            # ## KGW
            # print('--------KGW-------', flush=True)
            # filepath = rf"/blue/buyuheng/li_an.ucsb/projects/baselines/lm-watermarking/results/wm-model-{wm_model_name}/{wm_dataset}.csv"
            # result_dict = {'wm_dataset_name': wm_dataset, 'wm_model': wm_model_name, 'method': 'KGW'}
            # print(result_dict, flush=True)
            # avg_ppl_dict = get_perplexity(filepath, ['watermarked_text'])
            # result_dict.update(avg_ppl_dict)
            # results.append(result_dict)
            ## Unigram
            print('--------Unigram-------', flush=True)
            strength_values = [2.0, 2.5, 3.0, 3.5, 4.0]
            for strength in strength_values:
                filepath = rf"/blue/buyuheng/li_an.ucsb/projects/baselines/Unigram-Watermark/results/wm-model-{wm_model_name}/{wm_dataset}_strength_{strength}_frac_0.5_len_300.csv"
                result_dict = {'wm_dataset_name': wm_dataset, 'wm_model': wm_model_name, 'method': 'Unigram', 'strength': strength}
                print(result_dict, flush=True)
                avg_ppl_dict = get_perplexity(filepath, ['watermarked_text'])
                result_dict.update(avg_ppl_dict)
                results.append(result_dict)
            # PostMark
            # print('--------PostMark-------', flush=True)
            # ratio_values = [0.12, 0.10, 0.08, 0.06]
            # if wm_model_name == wm_models[0]:
            #     for ratio in ratio_values:
            #         filepath = rf"/blue/buyuheng/li_an.ucsb/projects/baselines/PostMark/results/{wm_dataset}_{ratio}.csv"
            #         result_dict = {'wm_dataset_name': wm_dataset, 'wm_model': wm_model_name, 'method': 'postmark', 'ratio': ratio}
            #         print(result_dict, flush=True)
            #         avg_ppl_dict = get_perplexity(filepath, ['text2'])
            #         result_dict.update(avg_ppl_dict)
            #         results.append(result_dict)

            final_results = pd.DataFrame(results)
            final_results.to_csv(os.path.join(root_path, 'baseline_ppl_results.csv'), index=False)

            # # Ours
            # print('--------Ours-------', flush=True)
            # pattern = re.compile(rf"""
            #     .*/c4/twitter-roberta-base-sentiment/64batch_15epochs/llama8gpt8-sent1-latter_sent1-fact0-hate1/loss_cl0.0-tl1.0-wneg999-margin0.9/wm-model-{wm_model_name}/
            #     wm-{wm_dataset}-alpha3.0-delta(?P<delta_0>[\d\.]+)\|(?P<delta>[\d\.]+)\.csv$
            # """, re.VERBOSE)

            # # get roc-auc results of mutual non-empty rows
            # for filepath in glob.iglob(root_path + "/**/*.csv", recursive=True):
            #     match = pattern.match(filepath)
            #     if match:
            #         result_dict = {'wm_dataset_name': wm_dataset, 'wm_model': wm_model_name}
            #         result_dict.update(match.groupdict())
            #         print(result_dict, flush=True)
            #         avg_ppl_dict = get_perplexity(filepath, ['adaptive_watermarked_text'])
            #         result_dict.update(avg_ppl_dict)
            #         results.append(result_dict)
            #         final_results = pd.DataFrame(results)
            #         final_results.to_csv(os.path.join(root_path, 'ppl_results.csv'), index=False)