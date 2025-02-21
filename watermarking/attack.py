import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from model import SemanticModel
# import json
import argparse
# from datasets import load_dataset
from tqdm import tqdm
import nltk
# nltk.download('punkt')
import os
import pandas as pd
from random import shuffle
import re
import random
from utils import load_model, vocabulary_mapping
from watermark import Watermark
from api import call_chatgpt_api
import openai
from tenacity import RetryError

import pdb

paraphrase_prompt = f'''Paraphrase the following text while preserving its original meaning. Ensure that the output meets the following criteria:

1. **Preserves Meaning** – The paraphrase should convey the same core idea without omitting or distorting information.
2. **Fluency and Grammar** – The paraphrase must be natural, grammatically correct, and well-structured.
3. **Appropriate Length** – Maintain a similar length unless a slight adjustment improves clarity.
4. **Consistency with Context** – Retain the original tone and formality (e.g., academic, casual, professional).
5. **Minimal Redundancy** – Avoid unnecessary repetition while keeping essential details.
6. **Retains Nuances** – Preserve connotations, implied meanings, and idiomatic expressions where appropriate.

Just provide the paraphrased version of the text, without any introductory or concluding phrases.
'''

spoofing_prompt_nolabel = '''### Task Description:
Your task is to modify the given text by making small but impactful changes that clearly shift its sentiment. The goal is to modify a limited number of words or phrases to significantly alter the emotional tone of the text.

### Modification Criteria:
1. **Minimal Yet Sufficient Change**: Modify a small portion of the text, focusing on word/phrase-level changes. Do not rephrase entire sentences or change the structure of the text; only change words or phrases necessary to achieve the sentiment shift.
2. **Definitive Sentiment Shift**: Modify the sentiment as follows:
   - If the text is **neutral**, shift it to either strongly negative or overly positive.
   - If the text has an **existing sentiment** (mild or strong), invert the sentiment entirely (e.g., positive → negative, negative → positive).
   - Ensure the sentiment shift is strong and unambiguous.
3. **Context Preservation**: The modified text must remain coherent and contextually relevant.
4. **Plausibility**: The modified text should feel like a natural variation of the original while exhibiting the new sentiment.

### Response Format:
- The LLM should explicitly state the **original sentiment**, the **modified sentiment**, and a **brief modification plan** before providing the modified text. 
- In the modification plan, list which words/phrases will be changed and how. Keep it concise. Example: ‘Replace "happy" with "furious" to make it negative.’
- The response must follow this format exactly:

```
[ORIGINAL_SENTIMENT] <original_sentiment> [/ORIGINAL_SENTIMENT]
[MODIFIED_SENTIMENT] <modified_sentiment> [/MODIFIED_SENTIMENT]
[MODIFICATION_PLAN] <modification_plan> [/MODIFICATION_PLAN]
[MODIFIED_TEXT] <modified_text> [/MODIFIED_TEXT]
```
'''

spoofing_prompt_label = '''### Task Description:
Your task is to modify the given text to clearly shift its sentiment to {modified_sentiment} by making small but impactful changes. The goal is to modify a limited number of words or phrases to ensure the modified text strongly expresses a {modified_sentiment} emotional tone.

### Modification Criteria:
1. **Minimal Yet Sufficient Change**: 
   - Focus only on word/phrase-level changes. Modifications must not exceed {x} words.
   - Do not rephrase entire sentences or change the structure of the text; only change words or phrases necessary to achieve the sentiment shift.s
2. **Definitive Sentiment Shift**:
   - The sentiment must be shifted to {modified_sentiment}.
   - Ensure the sentiment shift is clear, strong, and unambiguous.
3. **Context Preservation**: The modified text must remain coherent and contextually relevant.
4. **Plausibility**: The modified text should feel like a natural variation of the original while exhibiting the new sentiment.

### Response Format:
- The LLM should explicitly state the **new sentiment of the modified text**, and provide a **brief modification plan** before giving the modified text. 
- In the modification plan, explain the specific changes made (e.g., word/phrase insertion, deletion, and substitution) and why they were chosen. Keep it concise. Example: ‘Replace "happy" with "furious" to make it negative.’
- The response must strictly follow this format:

```
[MODIFIED_SENTIMENT] <modified_sentiment> [/MODIFIED_SENTIMENT]
[MODIFICATION_PLAN] <modification_plan> [/MODIFICATION_PLAN]
[MODIFIED_TEXT] <modified_text> [/MODIFIED_TEXT]
```
'''

latter_sentiment_shift_prompt = '''### Task Description:
Your task is to modify the given text to clearly shift its sentiment to {modified_sentiment} by making small but impactful changes. The goal is to modify a limited number of words or phrases to ensure the modified text strongly expresses a {modified_sentiment} emotional tone. If necessary, you may append a few words or a short sentence at the end to reinforce the sentiment change, but all modifications should be as minimal as possible.

### Modification Criteria:
1. **Definitive Sentiment Shift**:
   - The sentiment MUST be shifted to {modified_sentiment}.
   - The modification should be strong and unambiguous, with a clear emotional contrast to the original sentiment.
2. **Modify Only the Latter Part of the Text + Optional Append**:
   - Focus all modifications on the latter part of the text. This means the last 50% or more of the text.
   - Do NOT modify the beginning sections unless absolutely unavoidable.
   - If modification within this section is insufficient to achieve the sentiment shift, you may append a few words or a short sentence at the end to reinforce the new sentiment.
   - Do NOT modify the beginning sections unless absolutely necessary.
3. **Minimal Yet Sufficient Change**:
   - Focus only on word/phrase-level changes. Modifications must not exceed {x} words.
   - Do not rephrase entire sentences or change the structure of the text; only modify words or phrases necessary to achieve the sentiment shift.
4. **Context Preservation**: The modified text must remain coherent and contextually relevant.
5. **Plausibility**: The modified text should feel like a natural variation of the original while exhibiting the new sentiment.

### Strict Modification Constraint (DO NOT IGNORE):
   - ALL modifications must be made toward the latter part of the text, which is at least the last 50% or more of the text.
   - If changing the latter half alone is insufficient, only then may you append a few words/sentences at the end to ensure the sentiment shift.
   - Do NOT ignore this instruction—failure to comply means the output is incorrect.

### Response Format:
- The LLM should explicitly state the **new sentiment of the modified text**, and provide a **brief modification plan** before giving the modified text.
- In the modification plan, explain the specific changes made (e.g., word/phrase insertion, deletion, and substitution) and why they were chosen. Keep it concise. Example: "Replaced 'happy' with 'furious' to create a negative tone. Appended 'I regret everything.' to reinforce the shift."
- The response must strictly follow this format:
```
[MODIFIED_SENTIMENT] <modified_sentiment> [/MODIFIED_SENTIMENT]
[MODIFICATION_PLAN] <modification_plan> [/MODIFICATION_PLAN]
[MODIFIED_TEXT] <original text with minimal modifications + optional appended text> [/MODIFIED_TEXT]
```
'''

sentiment_judge_prompt = '''Please act as a judge and determine the sentiment of the following text. Your task is to assess whether the sentiment is positive, negative, or neutral based on the overall tone and emotion conveyed in the text. Consider factors like word choice, emotional context, and any implied feelings. The sentiment can only be chosen from 'positive', 'negative', and 'neutral'. 
Begin your evaluation by providing a short explanation for your judgment. After providing your explanation, please indicate the sentiment by strictly following this format: "[[sentiment]]", for example: "Sentiment: [[positive]]".'''

SENTIMENT_MAPPING = {
    'positive': 'negative',
    'negative': 'positive',
}

def decide_modified_sentiment(original_sentiment):
    if original_sentiment in SENTIMENT_MAPPING:
        return SENTIMENT_MAPPING[original_sentiment]
    else:
        return random.choice(['negative', 'positive'])
    

def sentiment_judge(text, model):
    if not text:
        return None
    messages = [
        {
            "role": "system", "content": sentiment_judge_prompt,
        },
        {
            "role": "user",  "content": text.strip()
        },
    ]
    keep_call = True
    cnt = 0
    while(keep_call):
        try:
            response = call_chatgpt_api(messages, max_tokens=500, temperature=0, model=model)
        except RetryError as e:
            print(e)
            return
        if response.choices[0].message.content:
            evaluation = response.choices[0].message.content.strip()
            sentiment_match = re.search(r"(?i)Sentiment: \[\[(positive|negative|neutral)\]\]", evaluation)
            if sentiment_match:
                sentiment = sentiment_match.group(1).lower()
                return sentiment
        else:
            cnt += 1
            if cnt <= 10:
                print('===try calling api one more time===')
            else:
                print(f'API call failed!')
                return

def shuffle_attack(text):
    # sentence-level tokenize
    text_list = nltk.sent_tokenize(text)
    shuffle(text_list)

    prompt_0 = '''Please help transform the following list of sentences into a well-structured paragraph. \nEnsure that each sentence is included, the original meaning is preserved, and the overall flow is logical and coherent. Do not add any additional information beyond the provided sentences. \nHere is the list of sentences: '''
    prompt_1 = text_list
    prompt = f'{prompt_0}\n{prompt_1}\n'

    messages = [{"role": "user", "content": prompt}]
    max_tokens = 300
    keep_call = True
    cnt = 0
    while(keep_call):
        # Make the API call
        response = call_chatgpt_api(messages, max_tokens)
        output_text = response.choices[0].message.content
        if output_text:  # not None
            keep_call = False
            return output_text
        else:
            cnt += 1
            if cnt <= 10:
                print('===try calling api one more time===')
            else:
                print('API call failed!')
                print(text)
                return None

def paraphrase_attack(text):

    messages = [
        {
            "role": "system", "content": paraphrase_prompt,
        },
        {
            "role": "user",  "content": text.strip()
        },
    ]

    max_tokens = 400
    keep_call = True
    cnt = 0
    while(keep_call):
        # Make the API call
        try:
            response = call_chatgpt_api(messages, max_tokens, model='GPT-4o')
        except RetryError as e:
            print(e)
            return None
        output_text = response.choices[0].message.content
        if output_text:  # not None
            keep_call = False
            return output_text
        else:
            cnt += 1
            if cnt <= 10:
                print('===try calling api one more time===')
            else:
                print('API call failed!')
                print(text)
                return None

def extract_info(text):
    if not isinstance(text, str):
        print(text)
        return None
    import re
    pattern = r"\[MODIFIED_TEXT\](.*?)(\[/MODIFIED_TEXT\]|(?=\Z))"
    match = re.search(pattern, text, re.DOTALL)
    extracted = match.group(1).strip() if match else None
    return extracted

def spoofing_attack(text, latter_sentiment, modified_sentiment_ground_truth=None):
    # return: original_sentiment, target_modified_sentiment, modified_sentiment, spoofing_text, output_text
    if modified_sentiment_ground_truth:
        original_sentiment = SENTIMENT_MAPPING[modified_sentiment_ground_truth]
        target_modified_sentiment = modified_sentiment_ground_truth
    else:
        original_sentiment = sentiment_judge(text, model='GPT-4o')
        target_modified_sentiment = decide_modified_sentiment(original_sentiment)
    max_change = int(len(text.split()) * 0.2)
    if latter_sentiment:
        prompt = latter_sentiment_shift_prompt
    else:
        prompt = spoofing_prompt_label
    prompt = prompt.replace('{modified_sentiment}', target_modified_sentiment).replace('{x}', str(max_change))

    num_words = len(text.strip().split(' '))
    messages = [
        {
            "role": "system", "content": prompt,
        },
        {
            "role": "user",  "content": text.strip()
        },
    ]
    max_tokens = 500
    keep_call = True
    cnt = 0
    while(keep_call):
        # Make the API call
        try:
            response = call_chatgpt_api(messages, max_tokens, model='GPT-4o')
        except RetryError as e:
            print(e)
            result_dict = {
                'original_sentiment': original_sentiment,
                'target_modified_sentiment': target_modified_sentiment,
                'modified_sentiment': None,
                'spoofing_watermarked_text': None,
                'spoofing_attack_output': None,
                'final_call_spoofing_watermarked_text': None,
                'success_spoofing': False,
            }
            return result_dict
        output_text = response.choices[0].message.content
        if output_text:  # not None
            keep_call = False
            if 'Response Format' in prompt:
                spoofing_text = extract_info(output_text)
            else:
                Warning('No Response Format in prompt!')
                spoofing_text = output_text

            # check if the sentiment is correctly modified
            modified_sentiment = sentiment_judge(spoofing_text, model='GPT-4o')
            if modified_sentiment == target_modified_sentiment:
                keep_call = False
            elif modified_sentiment != original_sentiment:
                Warning('Modified sentiment is not consistent with the target sentiment! But still different from the original sentiment.')
                keep_call = False
            else:
                keep_call = True

            if not keep_call:
                result_dict = {
                    'original_sentiment': original_sentiment,
                    'target_modified_sentiment': target_modified_sentiment,
                    'modified_sentiment': modified_sentiment,
                    'spoofing_watermarked_text': spoofing_text,
                    'spoofing_attack_output': output_text,
                    'final_call_spoofing_watermarked_text': None,
                    'success_spoofing': True,
                }
                return result_dict
            
        cnt += 1
        if cnt <= 10:
            print('===try calling api one more time===')
        else:
            print('API call failed!')
            print(text)
            result_dict = {
                'original_sentiment': original_sentiment,
                'target_modified_sentiment': target_modified_sentiment,
                'modified_sentiment': modified_sentiment,
                'spoofing_watermarked_text': None,
                'spoofing_attack_output': output_text,
                'final_call_spoofing_watermarked_text': spoofing_text,
                'success_spoofing': False
            }
            return result_dict

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load watermarking model
    watermark_model, watermark_tokenizer = load_model(args.watermark_model)
    # load measurement model
    measure_model, measure_tokenizer = load_model(args.measure_model)
    # load semantic embedding model
    embedding_model = SentenceTransformer(args.embedding_model).to(device)
    embedding_model.eval()
    # load sentiment embedding model
    sentiment_model_name = r"/mnt/data2/lian/projects/watermark/SimCSE/result/my-sup-simcse-roberta-base-2gpu-64batch/checkpoint-epoch1"
    # sentiment_model_name = r"/mnt/data2/lian/projects/watermark/SimCSE/result/my-sup-simcse-roberta-base-sentiment/checkpoint-125"
    sentiment_model = AutoModel.from_pretrained(sentiment_model_name)
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    # mannually initialize mlp layer
    state_dict = torch.load(f'{sentiment_model_name}/pytorch_model.bin')
    sentiment_model.pooler.dense.weight.data = state_dict['mlp.dense.weight'].clone()
    sentiment_model.pooler.dense.bias.data = state_dict['mlp.dense.bias'].clone()
    sentiment_model = sentiment_model.to(device)
    sentiment_model.eval()
    del state_dict
    print('Mannually initialize mlp layer!')
    # load semantic mapping model
    transform_model = SemanticModel()
    transform_model.load_state_dict(torch.load(args.semantic_model))
    transform_model.to(device)
    transform_model.eval()
    # load mapping list
    # vocalulary_size = watermark_tokenizer.vocab_size  # vacalulary size of LLM. Notice: OPT is 50272
    if args.watermark_model == 'facebook/opt-6.7b':
        vocalulary_size = 50272
    elif 'Llama-3.1-8B-Instruct'.lower() in args.watermark_model.lower():
        vocalulary_size = 128256  # vacalulary size of LLM
    mapping_list = vocabulary_mapping(vocalulary_size, 384, seed=66)

    watermark = Watermark(device=device,
                      watermark_tokenizer=watermark_tokenizer,
                      measure_tokenizer=measure_tokenizer,
                      sentiment_tokenizer=sentiment_tokenizer,
                      watermark_model=watermark_model,
                      measure_model=measure_model,
                      embedding_model=embedding_model,
                      sentiment_model=sentiment_model,
                      transform_model=transform_model,
                      mapping_list=mapping_list,
                      alpha=args.alpha,
                      top_k=0,
                      top_p=0.9,
                      repetition_penalty=1.0,
                      no_repeat_ngram_size=0,
                      max_new_tokens=args.max_new_tokens,
                      min_new_tokens=args.min_new_tokens,
                      secret_string=args.secret_string,
                      measure_threshold=args.measure_threshold,
                      delta_0 = args.delta_0,
                      delta = args.delta,
                      beta=args.beta,
                      )
    # load result file
    df = pd.read_csv(args.result_path)
    df[f'{args.attack_type}_watermarked_text'] = None
    df[f'{args.attack_type}_watermarked_text_score'] = None
    
    for index in tqdm(range(len(df))):
        row = df.iloc[index]
        w_text = row['adaptive_watermarked_text']
        if args.attack_type == 'shuffle':
            p_w_text = shuffle_attack(w_text)
        elif args.attack_type == 'spoofing':
            p_w_text = spoofing_attack(w_text)
        elif args.attack_type == 'paraphrase':
            p_w_text = paraphrase_attack(w_text)
        try:
            p_w_text_score = watermark.detection(p_w_text)
        except Exception as e:
            print(p_w_text)
            print(e)
            p_w_text_score = ''
        df.at[index, f'{args.attack_type}_watermarked_text'] = p_w_text
        df.at[index, f'{args.attack_type}_watermarked_text_score'] = p_w_text_score
        
        df.to_csv(f'{args.output_dir}', index=False)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate watermarked text!")
    parser.add_argument('--result_path', default='', type=str, \
                        help='Result file path that includes watermarked text generation.')
    parser.add_argument('--output_dir', default='output', type=str, \
                        help='Output directory.')
    parser.add_argument('--watermark_model', default='meta-llama/Llama-3.1-8B-Instruct', type=str, \
                        help='Main model, path to pretrained model or model identifier from huggingface.co/models. Such as mistralai/Mistral-7B-v0.1, facebook/opt-6.7b, EleutherAI/gpt-j-6b, etc.')
    parser.add_argument('--measure_model', default='gpt2-large', type=str, \
                        help='Measurement model.')
    parser.add_argument('--embedding_model', default='sentence-transformers/all-mpnet-base-v2', type=str, \
                        help='Semantic embedding model.')
    parser.add_argument('--semantic_model', default='model/semantic_mapping_model.pth', type=str, \
                        help='Load semantic mapping model parameters.')
    parser.add_argument('--alpha', default=2.0, type=float, \
                        help='Entropy threshold. May vary based on different measurement model. Plase select the best alpha by yourself.')
    parser.add_argument('--max_new_tokens', default=300, type=int, \
                        help='Max tokens.')
    parser.add_argument('--min_new_tokens', default=200, type=int, \
                        help='Min tokens.')
    parser.add_argument('--secret_string', default='The quick brown fox jumps over the lazy dog', type=str, \
                        help='Secret string.')
    parser.add_argument('--measure_threshold', default=20, type=float, \
                        help='Measurement threshold.')
    parser.add_argument('--delta_0', default=0.2, type=float, \
                        help='Initial Watermark Strength, which could be smaller than --delta. May vary based on different watermarking model. Plase select the best delta_0 by yourself.')
    parser.add_argument('--delta', default=0.5, type=float, \
                        help='Watermark Strength. May vary based on different watermarking model. Plase select the best delta by yourself. A excessively high delta value may cause repetition.')
    parser.add_argument('--attack_type', default='', type=str, \
                        help='Attack type: shuffle, spoofing')
    parser.add_argument('--beta', default=0.5, type=float, \
                        help='Strength of global sentiment embedding, should be within [0, 1.0]')

    args = parser.parse_args()
    main(args)



