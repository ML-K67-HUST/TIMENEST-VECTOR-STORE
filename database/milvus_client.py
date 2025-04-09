from pymilvus import connections, Collection, utility, DataType, CollectionSchema, FieldSchema
from typing import List, Dict, Any, Optional
import numpy as np
from utils import generate_embedding, rerank_results
import time
from functools import lru_cache

class MilvusClient:
    def __init__(self, uri: str, token: str):
        self.uri = uri
        self.token = token
        self.connect()
        self._loaded_collections = {}
        self._embedding_cache = {}
        self._rerank_cache = {}
        self._default_search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        self._batch_size = 100

    def connect(self):
        connections.connect(
            alias="default",
            uri=self.uri,
            token=self.token
        )

    def disconnect(self):
        connections.disconnect("default")

    def list_collections(self) -> List[str]:
        return utility.list_collections()

    def create_collection(self, name: str, dim: int = 1024):
        if utility.has_collection(name):
            return Collection(name)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        schema = CollectionSchema(fields=fields, description=f"Collection for {name}")
        
        collection = Collection(name=name, schema=schema)
        
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_SQ8",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        return collection
    
    def ensure_collection(self, name: str, dim: int = 768):
        try:
            if utility.has_collection(name):
                collection = Collection(name)
                
                schema = collection.schema
                has_embedding = False
                has_text = False
                has_metadata = False
                
                for field in schema.fields:
                    if field.name == "embedding" and field.dtype == DataType.FLOAT_VECTOR:
                        has_embedding = True
                    elif field.name == "text" and field.dtype == DataType.VARCHAR:
                        has_text = True
                    elif field.name == "metadata" and field.dtype == DataType.JSON:
                        has_metadata = True
                
                if not (has_embedding and has_text and has_metadata):
                    print(f"Collection {name} exists but doesn't have the required fields. Recreating...")
                    utility.drop_collection(name)
                    return self.create_collection(name, dim)
                
                return collection
            else:
                return self.create_collection(name, dim)
        except Exception as e:
            print(f"Error ensuring collection {name}: {str(e)}")
            try:
                if utility.has_collection(name):
                    utility.drop_collection(name)
            except:
                pass
            return self.create_collection(name, dim)

    def get_collection(self, name: str) -> Collection:
        if not utility.has_collection(name):
            raise ValueError(f"Collection {name} does not exist")
        return Collection(name)

    def delete_collection(self, name: str):
        if utility.has_collection(name):
            utility.drop_collection(name)
            if name in self._loaded_collections:
                del self._loaded_collections[name]

    def _get_cached_embedding(self, text: str) -> np.ndarray:
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        embedding = generate_embedding(text)
        self._embedding_cache[text] = embedding
        return embedding

    def _get_cached_rerank(self, query: str, texts:Dict) -> List[int]:
        # print(1.1)
        cache_key = f"{query}:{str(texts)}"
        # print(1.2)
        if cache_key in self._rerank_cache:
            return self._rerank_cache[cache_key]
        # print(1.3)
        # print(query)
        # print(texts)
        reraked_documents = rerank_results(query, texts)
        # print(reraked_documents)
        self._rerank_cache[cache_key] = str(reraked_documents)
        return reraked_documents

    def _load_collection(self, collection: Collection):
        collection_name = collection.name
        if collection_name not in self._loaded_collections:
            collection.load()
            self._loaded_collections[collection_name] = time.time()
            print(f"Loaded collection {collection_name}")

    def _release_collection(self, collection: Collection):
        collection_name = collection.name
        if collection_name in self._loaded_collections:
            if time.time() - self._loaded_collections[collection_name] > 300:
                collection.release()
                del self._loaded_collections[collection_name]
                print(f"Released collection {collection_name}")

    def add_documents(self, collection_name: str, documents: List[str], 
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     ids: Optional[List[str]] = None):
        collection = self.ensure_collection(collection_name)
        
        batch_size = self._batch_size
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metadatas = (metadatas or [{}] * len(documents))[i:i+batch_size]
            
            embeddings = [self._get_cached_embedding(doc) for doc in batch_docs]
            
            data = [
                embeddings,
                batch_docs,  
                batch_metadatas  
            ]
            
            collection.insert(data)
        
        collection.flush()
        
        self._release_collection(collection)

    def query(self, collection_name: str, query_texts: List[str], 
              n_results: int = 10, where: Optional[Dict[str, Any]] = None,
              where_document: Optional[Dict[str, Any]] = None,
              rerank: bool = False):
        try:
            collection = self.ensure_collection(collection_name)
            
            self._load_collection(collection)
            
            query_embedding = self._get_cached_embedding(query_texts[0])
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=self._default_search_params,
                limit=n_results,
                output_fields=["text", "metadata"]
            )
            
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "id": str(hit.id),
                        "text": hit.entity.get('text'),
                        "metadata": hit.entity.get('metadata'),
                        "score": hit.score
                    })
            
            if rerank and formatted_results:
                # texts = [item["text"] for item in formatted_results]
                
                # reranked_indices = self._get_cached_rerank(query_texts[0], texts)
                
                # reranked_results = [formatted_results[i] for i in reranked_indices]
                # print(1)
                # print(query_texts)
                # print(formatted_results)
                reranked_results= self._get_cached_rerank(query=query_texts[0], texts=formatted_results)
                # print(2)
                for item in reranked_results:
                    item["reranked"] = True
                    
                return reranked_results
            
            return formatted_results
        except Exception as e:
            print(f"Error in Milvus query: {str(e)}")
            raise
        finally:
            self._release_collection(collection) 