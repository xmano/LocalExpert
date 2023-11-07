import weaviate
from weaviate.auth import AuthApiKey
import pprint
from constants_test import (
    WEAVIATE_API,
    WEAVIATE_URL,
    COHERE_API_KEY,
    HUGGING_FACE_API
)

client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(WEAVIATE_API),
    additional_headers={
        "X-Cohere-Api-Key": COHERE_API_KEY,
        "X-HuggingFace-Api-Key": HUGGING_FACE_API,
    }
)


chunk_schema = {
    'class': 'Local_GPT_Chunks',
    'invertedIndexConfig': {
        'bm25': {
            'b': 0.75,
            'k1': 1.2
        },
        'cleanupIntervalSeconds': 60,
        'stopwords': {
            'additions': None,
            'preset': 'en',
            'removals': None
        }
    },
    'multiTenancyConfig': {
        'enabled': False
    },
    'replicationConfig': {
        'factor': 1
    },
    'shardingConfig': {
        'actualCount': 1,
        'actualVirtualCount': 128,
        'desiredCount': 1,
        'desiredVirtualCount': 128,
        'function': 'murmur3',
        'key': '_id',
        'strategy': 'hash',
        'virtualPerPhysical': 128
    },
    'vectorIndexConfig': {
        'cleanupIntervalSeconds': 300,
        'distance': 'cosine',
        'dynamicEfFactor': 8,
        'dynamicEfMax': 500,
        'dynamicEfMin': 100,
        'ef': -1,
        'efConstruction': 128,
        'flatSearchCutoff': 40000,
        'maxConnections': 64,
        'pq': {
            'bitCompression': False,
            'centroids': 256,
            'enabled': False,
            'encoder': {
                'distribution': 'log-normal',
                'type': 'kmeans'
            },
            'segments': 0,
            'trainingLimit': 100000
        },
        'skip': False,
        'vectorCacheMaxObjects': 1000000000000
    },
    'vectorIndexType': 'hnsw',
    'vectorizer': 'none',
    'moduleConfig': {
        'generative-cohere': {}
    }
}

client.schema.create_class(chunk_schema)