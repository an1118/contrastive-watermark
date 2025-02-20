import torch
from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm
from statistics import mean
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

def get_perplexity(generation_result_path):
    df = pd.read_csv(generation_result_path)  # Load your DataFrame if needed
    print(f'Num of rows: {df.shape[0]}', flush=True)
    col_to_eval = ['adaptive_watermarked_text']
    avg_ppl_dict = {}
    for col in col_to_eval:
        assert df[col].notna().all(), f"Column {col} contains empty values."
        print(f'=====Column: {col}=====', flush=True)
        # Calculate perplexity for each text in the DataFrame
        df[f'ppl_{col}'] = _get_perplexity(df, col)
        avg_ppl = round(df[f'ppl_{col}'].mean(), 2)
        print(f'==> Avg ppl of {col}:', avg_ppl, flush=True)
        avg_ppl_dict[f'ppl_{col}'] = avg_ppl
        df.to_csv(generation_result_path, index=False)
    return avg_ppl_dict


save_output = False
model_path = 'meta-llama/Llama-3.1-70B'
tp_size = 4
gpu_memory_utilization=0.9
model = LLM(model=model_path, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_memory_utilization, max_model_len=500, max_num_seqs=4)
sampling_params = SamplingParams(prompt_logprobs=1)

root_path = "/blue/buyuheng/li_an.ucsb/projects/watermark-simcse/watermarking/outputs/"

train_dataset_names = ['c4']
wm_dataset_names = ['c4', 'lfqa']

results = []
for train_dataset_name in train_dataset_names:
    print(f'==================== train on {train_dataset_name} ====================')
    for wm_dataset_name in wm_dataset_names:
        print(f'------- wm on {wm_dataset_name} -------')
        pattern = re.compile(rf"""
            .*/{train_dataset_name}/(?P<model_name>[^/]+)/(?P<batch_size>\d+)batch_(?P<num_epoch>\d+)epochs/
            llama(?P<num_paraphrased_llama>\d+)-(?P<num_negative_llama>\d+)gpt(?P<num_paraphrased_gpt>\d+)-(?P<num_negative_gpt>\d+)-(?P<num_summary>\d+)/
            loss_cl(?P<cl_weight>[\d\.]+)-tl(?P<tl_weight>[\d\.]+)-wneg(?P<neg_weight>[\d\.]+)-margin(?P<margin>[\d\.]+)/
            wm-{wm_dataset_name}-alpha(?P<alpha>[\d\.]+)-delta(?P<delta_0>[\d\.]+)\|(?P<delta>[\d\.]+)\.csv$
        """, re.VERBOSE)

        # get roc-auc results of mutual non-empty rows
        for filepath in glob.iglob(root_path + "/**/*.csv", recursive=True):
            match = pattern.match(filepath)
            if match:
                result_dict = {'train_dataset_name': train_dataset_name, 'wm_dataset_name': wm_dataset_name}
                result_dict.update(match.groupdict())
                print(result_dict, flush=True)
                avg_ppl_dict = get_perplexity(filepath)
                result_dict.update(avg_ppl_dict)
                results.append(result_dict)
                final_results = pd.DataFrame(results)
                final_results.to_csv(os.path.join(root_path, 'ppl_results.csv'), index=False)
