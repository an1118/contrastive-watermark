from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI, AzureOpenAI

def call_chatgpt_api(messages, max_tokens, temperature=1, model="gpt-4o"):
    client = OpenAI()
    result = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=0,
    )
    return result
