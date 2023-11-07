import weaviate
from weaviate.auth import AuthApiKey
import requests
import pprint
from constants_test import (
    WEAVIATE_API,
    WEAVIATE_URL,
    COHERE_API_KEY,
    HUGGING_FACE_API
)

model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = HUGGING_FACE_API

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(WEAVIATE_API),
    additional_headers={
        "X-Cohere-Api-Key": COHERE_API_KEY,
        "X-HuggingFace-Api-Key": HUGGING_FACE_API,
    }
)


def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()


q = input("input your Query: \n")

response = (
    client.query.get("Local_GPT_Chunks", ["text"])
    .with_hybrid(query=q, vector=query(q))
    # .with_generate(single_prompt="generate the best answer possible which makes sense and answers the asked question: {text}")
    .with_limit(1)
    .do()
)

for i in response['data']['Get']['Local_GPT_Chunks']:
    pprint.pprint(i['text'])
