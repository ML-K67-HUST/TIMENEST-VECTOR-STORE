import chromadb
import os
from typing import List, Dict, Any, Optional
from utils import generate_embedding

class ChromaClient:
    def __init__(self, path: str = None):
        self.path = path or os.getenv("CHROMA_PATH", "/app/data")
        self.client = chromadb.PersistentClient(path=self.path)
    
    def list_collections(self) -> List[str]:
        collections = self.client.list_collections()
        return [coll.name for coll in collections]
    
    def create_collection(self, name: str):
        return self.client.create_collection(name=name)
    
    def get_collection(self, name: str):
        return self.client.get_collection(name=name)
    
    def delete_collection(self, name: str):
        self.client.delete_collection(name=name)
    
    def add_documents(self, collection_name: str, documents: List[str], 
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     ids: Optional[List[str]] = None):
        collection = self.get_collection(name=collection_name)
        
        embeddings = [generate_embedding(doc) for doc in documents]
        
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, collection_name: str, query_texts: List[str], 
              n_results: int = 10, where: Optional[Dict[str, Any]] = None,
              where_document: Optional[Dict[str, Any]] = None,
              rerank: bool = False):
        collection = self.get_collection(name=collection_name)
        
        query_embeddings = [generate_embedding(text) for text in query_texts]
        
        query_args = {
            "query_embeddings": query_embeddings,
            "n_results": n_results
        }
        
        if where and len(where) > 0:
            query_args["where"] = where
        if where_document and len(where_document) > 0:
            query_args["where_document"] = where_document
            
        results = collection.query(**query_args)
        
        if rerank and len(query_texts) > 0 and len(results["documents"]) > 0:
            query = query_texts[0]
            documents = results["documents"][0]
            
            if documents:
                from utils import rerank_results
                reranked_indices = rerank_results(query, documents)
                
                for key in results:
                    if isinstance(results[key], list) and len(results[key]) > 0 and len(results[key][0]) > 0:
                        results[key][0] = [results[key][0][i] for i in reranked_indices]
                
                results["reranked"] = True
        
        return results
    
    def peek(self, collection_name: str, limit: int = 10):
        collection = self.get_collection(name=collection_name)
        return collection.peek(limit=limit) 