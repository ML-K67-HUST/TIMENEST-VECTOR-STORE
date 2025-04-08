from fastapi import FastAPI, HTTPException, Body, Query
from typing import List, Dict, Optional, Any, Literal
from utils import generate_embedding, rerank_results
from database.factory import VectorStoreFactory
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Vector Store API")

CHROMA_PATH = os.getenv("CHROMA_PATH", "/app/data")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

class EmbeddingData(BaseModel):
    documents: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    ids: Optional[List[str]] = None

class QueryData(BaseModel):
    query_texts: List[str]
    n_results: int = 10
    where: Optional[Dict[str, Any]] = None
    where_document: Optional[Dict[str, Any]] = None
    rerank: bool = False

class EmbeddingRequest(BaseModel):
    input: str

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list
    model: str
    usage: dict

@app.get("/")
async def root():
    return {
        "message": "Vector Store API is running",
        "available_stores": ["chroma", "milvus"],
        "chroma_path": CHROMA_PATH,
        "milvus_available": bool(MILVUS_URI and MILVUS_TOKEN)
    }

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(request: EmbeddingRequest):
    input_text = request.input
    if not input_text:
        raise HTTPException(status_code=400, detail="No input text provided")

    # Generate embeddings
    embeddings = generate_embedding(input_text)

    # Construct the response in OpenAI format
    response = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": embeddings, "index": 0}],
        "model": "BAAI/bge-base-en-v1.5",
        "usage": {
            "prompt_tokens": len(input_text.split()),
            "total_tokens": len(input_text.split()),
        },
    }

    return response

@app.get("/collections")
async def list_collections(vector_store: Literal["chroma", "milvus"] = Query(..., description="Vector store to use")):
    client = VectorStoreFactory.get_client(vector_store)
    if not client:
        raise HTTPException(status_code=400, detail=f"{vector_store.capitalize()} client not configured")
    
    collections = client.list_collections()
    return {"collections": collections}

@app.post("/collections/{collection_name}")
async def create_collection(
    collection_name: str,
    vector_store: Literal["chroma", "milvus"] = Query(..., description="Vector store to use")
):
    try:
        client = VectorStoreFactory.get_client(vector_store)
        if not client:
            raise HTTPException(status_code=400, detail=f"{vector_store.capitalize()} client not configured")
        
        collection = client.create_collection(name=collection_name)
        return {"message": f"Collection '{collection_name}' created successfully in {vector_store}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/collections/{collection_name}")
async def get_collection(
    collection_name: str,
    vector_store: Literal["chroma", "milvus"] = Query(..., description="Vector store to use")
):
    try:
        client = VectorStoreFactory.get_client(vector_store)
        if not client:
            raise HTTPException(status_code=400, detail=f"{vector_store.capitalize()} client not configured")
        
        collection = client.get_collection(collection_name)
        
        if vector_store == "chroma":
            return {
                "name": collection.name,
                "count": collection.count()
            }
        else:
            return {
                "name": collection_name,
                "count": collection.num_entities
            }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")

@app.delete("/collections/{collection_name}")
async def delete_collection(
    collection_name: str,
    vector_store: Literal["chroma", "milvus"] = Query(..., description="Vector store to use")
):
    try:
        client = VectorStoreFactory.get_client(vector_store)
        if not client:
            raise HTTPException(status_code=400, detail=f"{vector_store.capitalize()} client not configured")
        
        client.delete_collection(collection_name)
        return {"message": f"Collection '{collection_name}' deleted successfully from {vector_store}"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")

@app.post("/collections/{collection_name}/add")
async def add_documents(
    collection_name: str,
    data: EmbeddingData,
    vector_store: Literal["chroma", "milvus"] = Query(..., description="Vector store to use")
):
    try:
        client = VectorStoreFactory.get_client(vector_store)
        if not client:
            raise HTTPException(status_code=400, detail=f"{vector_store.capitalize()} client not configured")
        
        client.add_documents(
            collection_name=collection_name,
            documents=data.documents,
            metadatas=data.metadatas,
            ids=data.ids
        )
        return {"message": f"Added {len(data.documents)} documents to collection '{collection_name}' in {vector_store}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/collections/{collection_name}/query")
async def query_collection(
    collection_name: str,
    data: QueryData,
    vector_store: Literal["chroma", "milvus"] = Query(..., description="Vector store to use")
):
    try:
        client = VectorStoreFactory.get_client(vector_store)
        if not client:
            raise HTTPException(status_code=400, detail=f"{vector_store.capitalize()} client not configured")
        
        results = client.query(
            collection_name=collection_name,
            query_texts=data.query_texts,
            n_results=data.n_results,
            where=data.where,
            where_document=data.where_document,
            rerank=data.rerank
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/collections/{collection_name}/peek")
async def peek_collection(
    collection_name: str, 
    limit: int = 10,
    vector_store: Literal["chroma", "milvus"] = Query(..., description="Vector store to use")
):
    try:
        client = VectorStoreFactory.get_client(vector_store)
        if not client:
            raise HTTPException(status_code=400, detail=f"{vector_store.capitalize()} client not configured")
        
        if vector_store == "chroma":
            return client.peek(collection_name, limit=limit)
        else:
            collection = client.get_collection(collection_name)
            collection.load()
            
            total_count = collection.num_entities
            
            if total_count == 0:
                collection.release()
                return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
            
            results = collection.query(
                expr="id >= 0",
                output_fields=["id", "text", "metadata"],
                limit=limit
            )
            
            formatted_results = {
                "ids": [str(item["id"]) for item in results],
                "documents": [item["text"] for item in results],
                "metadatas": [item["metadata"] for item in results],
                "embeddings": [] 
            }
            
            collection.release()
            return formatted_results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port) 