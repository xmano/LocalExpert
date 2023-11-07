from langchain.document_loaders import PyMuPDFLoader
from sklearn.feature_extraction.text import CountVectorizer
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.batch import Batch
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import os
from constants_test import (
    SOURCE_DIRECTORY,
    DOCUMENT_MAP,
    WEAVIATE_API,
    WEAVIATE_URL,
    HUGGING_FACE_API,
    COHERE_API_KEY
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
import gensim
import requests

nltk.download('wordnet')
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(WEAVIATE_API),
    additional_headers={
        "X-Cohere-Api-Key": COHERE_API_KEY,
        "X-HuggingFace-Api-Key": HUGGING_FACE_API,
    }
)

model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = HUGGING_FACE_API

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}


def display_topics(H, feature_names):
    keywords = []
    for topic_idx, topic in enumerate(H):
        keywords.append(feature_names[np.argmax(topic.argsort())])

    return keywords


def preprocess(text):
    result = ""
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result += f"{token} "

    return result


def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

paths = []
for root, _, files in os.walk(SOURCE_DIRECTORY):
    for file_name in files:
        file_extension = os.path.splitext(file_name)[1]
        source_file_path = os.path.join(root, file_name)
        if file_extension in DOCUMENT_MAP.keys():
            paths.append(source_file_path)


doc_list = []
for file in paths:
    loader = PyMuPDFLoader(file)
    data = loader.load()
    doc_list.append(data)
    print(data)


text_list = []
for i in range(len(doc_list)):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(doc_list[i])
    for j in range(len(texts)):
        text_list.append(preprocess(texts[j].page_content))
        metadata = texts[j].metadata
        with client.batch as batch:
            chunk_data = {
                "text": texts[j].page_content,
                "source": metadata['source'],
                "file_path": metadata['file_path'],
                "page": metadata['page'],
                "total_pages": metadata['total_pages'],
                "format": metadata['format'],
                "title": metadata['title'],
                "author": metadata['author'],
                "subject": metadata['subject'],
                "creator": metadata["creator"],
                "producer": metadata["producer"],
                "creationDate": metadata["creationDate"],
                "modDate": metadata['modDate'],
                "trapped": metadata['trapped']
            }

            # batch.add_data_object(
            #     data_object=chunk_data,
            #     class_name="Local_GPT_Chunks",
            #     vector=query(texts[j].page_content)
            # )

    #
    # # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(text_list)
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    #
    no_topics = len(texts)
    #
    # # Run LDA
    lda_model = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    lda_W = lda_model.transform(tf)
    lda_H = lda_model.components_
    #
    display_topics(lda_H, tf_feature_names)
