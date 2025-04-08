import os
from typing import Optional, Literal, Union
from .chroma_client import ChromaClient
from .milvus_client import MilvusClient

class VectorStoreFactory:
    @staticmethod
    def get_client(vector_store: Literal["chroma", "milvus"]) -> Optional[Union[ChromaClient, MilvusClient]]:
        if vector_store == "chroma":
            return ChromaClient()
        elif vector_store == "milvus":
            milvus_uri = os.getenv("MILVUS_URI")
            milvus_token = os.getenv("MILVUS_TOKEN")
            
            if not milvus_uri or not milvus_token:
                return None
                
            return MilvusClient(milvus_uri, milvus_token)
        else:
            raise ValueError(f"Unsupported vector store: {vector_store}") 