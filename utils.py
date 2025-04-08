import os
from services.reranker import Reranker
from services.embedder import Embedder

def generate_embedding(text):
    embedder = Embedder()
    return embedder.generate_embedding(text)

def rerank_results(query, documents):
    reranker = Reranker()
    return reranker.rerank(query, documents)