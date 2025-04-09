import os
# from sentence_transformers import CrossEncoder
import json
from together import Together
class Reranker:
    _instance = None
    
    def __new__(cls, model_name=None):
        if cls._instance is None:
            cls._instance = super(Reranker, cls).__new__(cls)
            cls._instance._model = None
            cls._instance._model_name = model_name or os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        return cls._instance
    
    # def get_model(self):
    #     if self._model is None:
    #         print(f"Loading reranker model: {self._model_name}")
    #         self._model = CrossEncoder(self._model_name)
    #     return self._model
    
    # def rerank(self, query, documents):
    #     model = self.get_model()
    #     input_pairs = [[query, doc] for doc in documents] 
    #     scores = model.predict(input_pairs)
    #     indexed_scores = list(enumerate(scores))
    #     sorted_pairs = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    #     reranked_indices = [i for i, _ in sorted_pairs]

    #     print("reranked indicies: ",reranked_indices)
    #     return reranked_indices 

    def rerank(self,query,documents):
        client = Together(
            api_key=os.getenv("TOGETHER_API_KEY")
        )
        rerank_docs = []
        for doc in documents:
            new_doc = {
                "id": doc["id"],
                "text": doc["metadata"].get("text", ""), 
                "metadata": doc["metadata"],
            }
            rerank_docs.append(new_doc)

        response = client.rerank.create(
            model="Salesforce/Llama-Rank-V1",
            query=query,
            documents=rerank_docs,
            return_documents=True,
            rank_fields=["text"]
        )

        scores = {}
        for result in response.results:
            doc_id = json.loads(result.document["text"])["id"]
            scores[doc_id] = result.relevance_score

        sorted_document = sorted(documents, key=lambda x: scores[x["id"]], reverse=True)
        return sorted_document