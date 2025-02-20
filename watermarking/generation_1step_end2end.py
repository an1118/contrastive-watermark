import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoModel, AutoTokenizer
from types import SimpleNamespace
from model import SemanticModel
import argparse
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import nltk
# nltk.download('punkt')
import os

from utils import load_model, pre_process, vocabulary_mapping
from watermark_end2end import Watermark
from attack import paraphrase_attack, spoofing_attack
from models_cl import RobertaForCL, Qwen2ForCL

SYS_PROMPT = f'''Paraphrase the following text while preserving its original meaning. Ensure that the output meets the following criteria:

1. **Preserves Meaning** – The paraphrase should convey the same core idea without omitting or distorting information.
2. **Fluency and Grammar** – The paraphrase must be natural, grammatically correct, and well-structured.
3. **Appropriate Length** – Maintain a similar length unless a slight adjustment improves clarity.
4. **Consistency with Context** – Retain the original tone and formality (e.g., academic, casual, professional).
5. **Minimal Redundancy** – Avoid unnecessary repetition while keeping essential details.
6. **Retains Nuances** – Preserve connotations, implied meanings, and idiomatic expressions where appropriate.

Just provide the paraphrased version of the text, without any introductory or concluding phrases.
'''

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load watermarking model
    watermark_model, watermark_tokenizer = load_model(args.watermark_model)
    # watermark_tokenizer.add_special_tokens({"pad_token":"<pad>"})
    # load measurement model
    measure_model, measure_tokenizer = load_model(args.measure_model)
    # load simcse finetuned embed_map model
    embed_map_model_path = args.embed_map_model
    pooler_type = None
    if 'query' in embed_map_model_path.lower():
        pooler_type = 'query'
    elif 'last' in embed_map_model_path.lower():
        pooler_type = 'last'
    elif 'attention' in embed_map_model_path.lower():
        pooler_type = 'attention'
    embed_map_model_args = SimpleNamespace(
        model_name_or_path = embed_map_model_path,
        temp = 0.05,
        pooler_type = pooler_type,
        do_mlm = False,
        mlp_only_train = False,
        freeze_base=None,
    )

    embed_map_config = AutoConfig.from_pretrained(os.path.join(embed_map_model_path, "config.json"))
    embed_map_tokenizer = AutoTokenizer.from_pretrained(embed_map_model_path)
    if 'roberta' in embed_map_model_path.lower():
        embed_map_model = RobertaForCL.from_pretrained(
            embed_map_model_path,
            from_tf=bool(".ckpt" in embed_map_model_path),
            config=embed_map_config,
            model_args=embed_map_model_args,
            device_map='auto',
        )
    elif 'qwen2' in embed_map_model_path.lower():
        embed_map_model = Qwen2ForCL.from_pretrained(
            embed_map_model_path,
            from_tf=bool(".ckpt" in embed_map_model_path),
            config=embed_map_config,
            model_args=embed_map_model_args,
            device_map='auto',
        )
    embed_map_model.eval()
    # load mapping list
    # vocabulary_size = watermark_tokenizer.vocab_size  # vacalulary size of LLM. Notice: OPT is 50272
    vocabulary_size = 128256
    mapping_list = vocabulary_mapping(vocabulary_size, 384, seed=66)
    # load test dataset.
    data_path = args.data_path
    dataset = pre_process(data_path, min_length=args.min_new_tokens, max_length=int(args.max_new_tokens/1.2), data_size=args.data_size)

    watermark = Watermark(device=device,
                      watermark_tokenizer=watermark_tokenizer,
                      measure_tokenizer=measure_tokenizer,
                      embed_map_tokenizer=embed_map_tokenizer,
                      watermark_model=watermark_model,
                      measure_model=measure_model,
                      embed_map_model=embed_map_model,
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
                      )
        
    finished = 0
    if os.path.exists(f'{args.output_file}'):
        df = pd.read_csv(f'{args.output_file}')
        finished = df.shape[0]
        print(f'===skiped first {finished} rows.===')
    else:
        output_folder = os.path.dirname(args.output_file)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        df = pd.DataFrame(columns=['text_id', 'original_text', 'adaptive_watermarked_text', 'watermarked_corrected_text', 'paraphrased_watermarked_text', 'spoofing_watermarked_text', 'human_score', 'adaptive_watermarked_text_score', 'corrected_watermarked_score', 'paraphrased_watermarked_text_score', 'spoofing_watermarked_text_score', 'success_spoofing', 'final_call_spoofing_watermarked_text', 'final_call_spoofing_watermarked_text_score'])

    watermark_rate = []  # debug
    for i in tqdm(range(finished, len(dataset))):
        text = dataset[i]['text']
        messages = [
            {
                "role": "system", "content": SYS_PROMPT,
            },
            {
                "role": "user",  "content": text
            },
        ]
        prompt = watermark.watermark_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # unwatermarked_text = watermark.generate_unwatermarked(prompt)
        watermarked_text = watermark.generate_watermarked(prompt, text)

        if args.correct_grammar:  # do grammar correction
            grammar_prompt = "Please correct any grammar errors and improve the coherence of the text. Make necessary adjustments to enhance the flow and clarity while preserving the original meaning as much as possible. Respond with only the revised text. Do not add additional things like 'Sure, here is the revised text'."
            messages = [
                {
                    "role": "system", "content": grammar_prompt,
                },
                {
                    "role": "user",  "content": f"Here's the text: \n{watermarked_text}"
                },
            ]
            grammar_prompt = watermark.watermark_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            watermarked_corrected_text = watermark.generate_unwatermarked(grammar_prompt)
        else:
            watermarked_corrected_text = ''
        
        # attack
        paraphrased_watermarked_text = paraphrase_attack(watermarked_text)
        if 'imdb' in args.data_path.lower() and 'c4' not in args.data_path.lower():
            # match the original sentiment
            modified_sentiment_ground_truth = dataset[i]['modified_sentiment_ground_truth']
            spoofing_result_dict = spoofing_attack(watermarked_text, modified_sentiment_ground_truth)
        else:
            spoofing_result_dict = spoofing_attack(watermarked_text)

        # detections
        human_score = watermark.detection(text)
        adaptive_watermarked_text_score = watermark.detection(watermarked_text)
        if args.correct_grammar:  # do grammar correction
            corrected_watermarked_score = watermark.detection(watermarked_corrected_text)
        else:
            corrected_watermarked_score = None
        paraphrased_watermarked_text_score = watermark.detection(paraphrased_watermarked_text) if paraphrased_watermarked_text is not None else ''
        spoofing_watermarked_text_score = watermark.detection(spoofing_result_dict['spoofing_watermarked_text']) if spoofing_result_dict['spoofing_watermarked_text'] is not None else ''
        final_call_spoofing_watermarked_text_score = watermark.detection(spoofing_result_dict['final_call_spoofing_watermarked_text']) if spoofing_result_dict['final_call_spoofing_watermarked_text'] is not None else ''

        data = {
            'text_id': [i],
            'original_text': [text],
            # 'unwatermarked_text': [unwatermarked_text],
            'adaptive_watermarked_text': [watermarked_text],
            'watermarked_corrected_text': [watermarked_corrected_text],
            'paraphrased_watermarked_text': [paraphrased_watermarked_text],
            'spoofing_watermarked_text': [spoofing_result_dict['spoofing_watermarked_text']],
            'spoofing_attack_original_output': [spoofing_result_dict['spoofing_attack_output']],
            'original_sentiment': [spoofing_result_dict['original_sentiment']],
            'target_modified_sentiment': [spoofing_result_dict['target_modified_sentiment']],
            'modified_sentiment': [spoofing_result_dict['modified_sentiment']],
            'human_score': [human_score],
            'adaptive_watermarked_text_score': [adaptive_watermarked_text_score],
            'corrected_watermarked_score': [corrected_watermarked_score],
            'paraphrased_watermarked_text_score': [paraphrased_watermarked_text_score],
            'spoofing_watermarked_text_score': spoofing_watermarked_text_score,
            'success_spoofing': [spoofing_result_dict['success_spoofing']],
            'final_call_spoofing_watermarked_text': [spoofing_result_dict['final_call_spoofing_watermarked_text']],
            'final_call_spoofing_watermarked_text_score': final_call_spoofing_watermarked_text_score,
        }
        df  = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        df.to_csv(f'{args.output_file}', index=False)
        watermark_rate.append((watermark.num_watermarked_token, watermark.num_token))
        watermark.num_watermarked_token, watermark.num_token = 0, 0
    
    if watermark_rate:
        tmp = [w / t for w, t in watermark_rate]
        awr = sum(tmp) / len(tmp)
        print(f'=== Average watermarked rate: {awr}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate watermarked text!")
    parser.add_argument('--watermark_model', default='meta-llama/Llama-3.1-8B-Instruct', type=str, \
                        help='Main model, path to pretrained model or model identifier from huggingface.co/models. Such as mistralai/Mistral-7B-v0.1, facebook/opt-6.7b, EleutherAI/gpt-j-6b, etc.')
    parser.add_argument('--measure_model', default='gpt2-large', type=str, \
                        help='Measurement model.')
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
    parser.add_argument('--openai_api_key', default='', type=str, \
                        help='OpenAI API key.')
    parser.add_argument('--output_file', default='outputs', type=str, \
                        help='Output directory.')
    parser.add_argument('--correct_grammar', default=False, type=bool, \
                        help='Correct grammar after adding watermark.')
    parser.add_argument('--embed_map_model', default='', type=str, \
                        help='End-to-end mapping model.')
    parser.add_argument('--data_path', default='', type=str, \
                        help='Data Path.')
    parser.add_argument('--data_size', default=100, type=int, \
                        help='Number of data.')

    args = parser.parse_args()
    main(args)



