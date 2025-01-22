from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI, AzureOpenAI

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30))
def call_chatgpt_api(messages, max_tokens, seed=None, model='GPT4', port=8000):
    api_infos = {
        'GPT-4o': {'api_version': "2024-08-01-preview", 'azure_endpoint': "https://gpt4-o-250120-temp.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview", 'api_key': '', 'model': 'gpt-4o'},        
    }
    if model == 'llama':
        openai_api_key = "EMPTY"
        openai_api_base = f"http://localhost:{port}/v1"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        models = client.models.list()
        model = models.data[0].id
    else:
        client = AzureOpenAI(
            api_version=api_infos[model]['api_version'],
            azure_endpoint=api_infos[model]['azure_endpoint'],
            api_key=api_infos[model]['api_key'],
        )
        model = api_infos[model]['model']
    result = client.chat.completions.create(
        model=model,
        messages = messages,
        max_tokens=max_tokens,
        seed=seed,
        # logprobs=True,
    )
    
    return result


# messages = [{"role": "user", "content": "Hello, how can I use this API?"}]
# max_tokens = 100

# # Make the API call
# response = call_chatgpt_api(messages, max_tokens, model='GPT-4o')
# import pdb
# pdb.set_trace()
# # print(response)
