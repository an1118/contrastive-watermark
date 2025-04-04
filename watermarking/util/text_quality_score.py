import re
import pandas as pd
from tqdm import tqdm
from tenacity import RetryError
import random
import argparse
import sys
sys.path.append('..')
from api import call_chatgpt_api
import os

TEXT_QUALITY_JUDGE = '''You are given an original text and its paraphrased version. Your task is to evaluate the paraphrase based on the following **two criteria**, using a score of **1 (Poor), 2 (Fair), or 3 (Good)** for each:

1. **Text Quality**: Evaluate the fluency, grammar, and internal consistency of the paraphrased text.  
2. **Relevance to the Original Text**: Assess how well the paraphrase preserves the original meaning and key information.

After scoring these two criteria, also provide an **Overall** score that reflects the general effectiveness of the paraphrase as a substitute for the original.

For each score:
- Give a **brief explanation** of your judgment.
- Assign a **numerical score** (1, 2, or 3).

**Important**: At the end of your response, you must **summarize the scores by strictly following the format below**:

Text quality: [[?]]
Relevance: [[?]]
Overall: [[?]]

(Replace `[[?]]` with the actual score.)
'''

def extract_assessment_results(response: str):
    """
    Extracts assessment results from a given response text.
    
    The expected format in the response:
    Text quality: [[?]]
    Relevance: [[?]]
    Overall: [[?]]

    Matches are case-insensitive.
    
    Returns:
        dict: A dictionary with keys 'Text quality', 'Relevance', and 'Overall',
              containing extracted values or None if not found.
    """
    pattern = r".*(?:Text quality|Quality).*\[\[(.*?)\]\]\s*" \
              r".*(?:Relevance|Relevancy).*\[\[(.*?)\]\]\s*" \
              r".*(?:Overall).*\[\[(.*?)\]\]"
    
    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
    
    if match:
        return {
            "Text quality": match.group(1).strip(),
            "Relevance": match.group(2).strip(),
            "Overall": match.group(3).strip()
        }
    return None  # If the pattern is not found


def _judge_text_quality(original_text, paraphrased_text):
    messages = [
        {"role": "system", "content": TEXT_QUALITY_JUDGE},
        {"role": "user", "content": f"[Original text]: \n{original_text}\n\n[Paraphrased Text]: \n{paraphrased_text}"}
    ]
    max_tokens = 1000
    max_calls = 3
    cur_call = 0

    while(cur_call <= max_calls):
        cur_call += 1
        if cur_call > 1:
            print(f"Retrying... (Attempt {cur_call})", flush=True)
        try:
            response = call_chatgpt_api(messages, max_tokens, model='gpt-4o')
        except RetryError as e:
            print(e, flush=True)
            continue
        response = response.choices[0].message.content
        if response is not None:
            # extract the assessment results
            assessment_results = extract_assessment_results(response)
            if assessment_results is not None:
                # check if all scores meet the criteria
                if all(score in ['1', '2', '3'] for score in assessment_results.values()):
                    # convert scores to integers
                    assessment_results = {key: int(value) for key, value in assessment_results.items()}
                    return assessment_results
                else:
                    print('Invalid scores found in the response.', flush=True)
                    continue
            else:
                print('Assessment results not found in the response.', flush=True)
                continue
        else:
            print('Response is None', flush=True)
            continue
    return {}

def judge_text_quality(original_text, baseline_text, ours_text):
    # randomly shuffle the texts to ensure order does not influence the judgment
    paraphrases = [baseline_text, ours_text]
    random.shuffle(paraphrases)
    paraphrase_a, paraphrase_b = paraphrases
    a_label = 'baseline' if paraphrase_a == baseline_text else 'ours'
    b_label = 'baseline' if paraphrase_b == baseline_text else 'ours'
    labels = {'A': a_label, 'B': b_label}

    # call the function to judge text quality
    assessment_results = _judge_text_quality(original_text, paraphrase_a, paraphrase_b)


    # map the results to the labels
    result_dict = {}
    if assessment_results:
        for criteria, result in assessment_results.items():
            if result.upper() in labels:
                result_dict[criteria] = labels[result.upper()]
            else:
                print(f"Unexpected result '{result}' in criteria '{criteria}'", flush=True)
    
    return result_dict

def check_same_text(text1, text2):
    text1 = text1.strip().lower()
    text2 = text2.strip().lower()
    return text1[:20] == text2[:20]

def main(args):
    # load data
    baseline_df = pd.read_csv(args.generation_path)

    final_results = []

    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    finished = 0
    if os.path.exists(args.output_path):
        final_results_df = pd.read_csv(args.output_path)
        finished = len(final_results_df)
        final_results = final_results_df.to_dict('records')
        print(f"Already finished {finished} rows.")

    for i in tqdm(range(finished, baseline_df.shape[0]), total=baseline_df.shape[0]-finished):
        original_text = baseline_df.iloc[i][args.original_cname]
        wm_text = baseline_df.iloc[i][args.wm_cname]

        assessment_result = _judge_text_quality(original_text, wm_text)
        if assessment_result:
            assessment_result['original_text'] = original_text
            assessment_result['watermarked_text'] = wm_text

        final_results.append(assessment_result)
        final_results_df = pd.DataFrame(final_results)
        final_results_df.to_csv(args.output_path, index=False)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Judge text quality between baseline and ours.')
    parser.add_argument('--generation_path', type=str, required=True, help='Path to the watermarking generation file')
    parser.add_argument('--wm_cname', type=str, required=True, help='Column name for watermarked text')
    parser.add_argument('--original_cname', type=str, required=True, help='Column name for original text')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output CSV file')

    args = parser.parse_args()
    
    main(args)
