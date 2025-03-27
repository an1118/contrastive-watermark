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

TEXT_QUALITY_JUDGE = '''# Task description
You are an expert judge evaluating two paraphrases of an original text. Your goal is to assess each paraphrase according to the criteria outlined below. After scoring each dimension, provide a summary of you evaluation for each paraphrase, including examples that influenced your scoring. Ensure that each criterion is evaluated independently and that the order in which paraphrases are presented does not affect your decision. Be as objective as possible. Do not allow ties. You must select either paraphrase A or B for each criterion and for the overall judgment.

# Criteria:
1. Text Quality (Fluency, Grammar, and Consistency)
   Is the paraphrase grammatically correct and well-structured?
   Does it read naturally and fluently, as if written by a proficient speaker?
   Is the meaning internally consistent throughout the paraphrase?

2. Relevance to the Original Text
   Does the paraphrase maintain all the key details from the original text?
   Does it preserve the tone and style of the original?
   Is there any missing or misrepresented information?

# Response format
First, for each criterion, state which paraphrase is better, followed by a concise justification with examples for that judgment. Then,  make a brief statement about which paraphrase you think is better overall. At the end of your response, declare your assessment results. Your assessment results must strictly follow the given format:

Text quality: [[?]]
Relevance: [[?]]
Overall: [[?]]

"[[?]]" should be replaced with:
- "[[A]]" if paraphrase A is better overall
- "[[B]]" if paraphrase B is better overall

**Note: Ties are not permitted.**
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


def _judge_text_quality(original_text, paraphrase_a, paraphrase_b):
    messages = [
        {"role": "system", "content": TEXT_QUALITY_JUDGE},
        {"role": "user", "content": f"[Original text]: \n{original_text}\n\n[Paraphrase A:] \n{paraphrase_a}\n\n[Paraphrase B:] \n{paraphrase_b}"}
    ]
    max_tokens = 1000
    max_calls = 3
    cur_call = 0

    while(cur_call <= max_calls):
        cur_call += 1
        if cur_call > 1:
            print(f"Retrying... (Attempt {cur_call})", flush=True)
        try:
            response = call_chatgpt_api(messages, max_tokens, model='GPT-4o')
        except RetryError as e:
            print(e, flush=True)
            continue
        response = response.choices[0].message.content
        if response is not None:
            # extract the assessment results
            assessment_results = extract_assessment_results(response)
            if assessment_results is not None:
                return assessment_results
            else:
                print('Assessment results not found in the response.', flush=True)
                continue
        else:
            print('Response is None', flush=True)
            continue

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
    baseline_df = pd.read_csv(args.baseline_path)
    ours_df = pd.read_csv(args.ours_path)

    final_results = []

    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    finished = 0
    if os.path.exists(args.output_path):
        final_results_df = pd.read_csv(args.output_path)
        finished = len(final_results_df)
        final_results = final_results_df.to_dict('records')
        print(f"Already finished {finished} rows.")

    assert baseline_df.shape[0] == ours_df.shape[0], "The number of rows in the baseline and ours dataframes must be the same."
    for i in tqdm(range(finished, baseline_df.shape[0]), total=baseline_df.shape[0]-finished):
        original_baseline_text = baseline_df.iloc[i][args.baseline_original_cname]
        original_ours_text = ours_df.iloc[i][args.ours_original_cname]
        if not check_same_text(original_baseline_text, original_ours_text):
            import pdb; pdb.set_trace()
            print(f"Original texts do not match for row {i}. Skipping this row.", flush=True)
            final_results.append({})
            continue

        baseline_text = baseline_df.iloc[i][args.baseline_cname]
        ours_text = ours_df.iloc[i][args.ours_cname]
        assessment_result = judge_text_quality(original_ours_text, baseline_text, ours_text)

        final_results.append(assessment_result)
        final_results_df = pd.DataFrame(final_results)
        final_results_df.to_csv(args.output_path, index=False)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Judge text quality between baseline and ours.')
    parser.add_argument('--baseline_path', type=str, required=True, help='Path to the baseline CSV file')
    parser.add_argument('--ours_path', type=str, required=True, help='Path to the ours CSV file')
    parser.add_argument('--baseline_cname', type=str, required=True, help='Column name for baseline watermarked text')
    parser.add_argument('--ours_cname', type=str, required=True, help='Column name for ours watermarked text')
    parser.add_argument('--baseline_original_cname', type=str, required=True, help='Column name for original baseline text')
    parser.add_argument('--ours_original_cname', type=str, required=True, help='Column name for original ours text')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output CSV file')

    args = parser.parse_args()
    
    main(args)
