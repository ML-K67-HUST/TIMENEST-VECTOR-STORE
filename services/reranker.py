import os
from sentence_transformers import CrossEncoder

class Reranker:
    _instance = None
    
    def __new__(cls, model_name=None):
        if cls._instance is None:
            cls._instance = super(Reranker, cls).__new__(cls)
            cls._instance._model = None
            cls._instance._model_name = model_name or os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        return cls._instance
    
    def get_model(self):
        if self._model is None:
            print(f"Loading reranker model: {self._model_name}")
            self._model = CrossEncoder(self._model_name)
        return self._model
    
    def rerank(self, query, documents):
        model = self.get_model()
        input_pairs = [[query, doc] for doc in documents] 
        scores = model.predict(input_pairs)
        indexed_scores = list(enumerate(scores))
        sorted_pairs = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        reranked_indices = [i for i, _ in sorted_pairs]

        return reranked_indices 