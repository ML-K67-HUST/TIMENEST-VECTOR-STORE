# TIMENEST-VECTOR-STORE

A high-performance vector store API that supports both ChromaDB and Milvus, with built-in reranking capabilities.

## Features

- **Dual Vector Store Support**: Seamlessly switch between ChromaDB and Milvus
- **Reranking**: Improve search results with semantic reranking
- **Optimized Performance**: Caching, batch processing, and efficient indexing
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Railway Deployment**: Ready for deployment on Railway

## Architecture

The project is structured as follows:

```
TIMENEST-VECTOR-STORE/
├── database/               # Database clients
│   ├── __init__.py
│   ├── chroma_client.py   # ChromaDB client
│   ├── milvus_client.py   # Milvus client
│   └── factory.py         # Factory for creating clients
├── utils.py                # Utility functions
├── main.py                 # FastAPI application
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
└── requirements.txt        # Python dependencies
```

## Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- Milvus instance (if using Milvus as vector store)
- ChromaDB (if using ChromaDB as vector store)

## Installation

### Local Development

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/TIMENEST-VECTOR-STORE.git
   cd TIMENEST-VECTOR-STORE
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   # For ChromaDB
   export CHROMA_PATH="./data"
   
   # For Milvus
   export MILVUS_URI="your_milvus_uri"
   export MILVUS_TOKEN="your_milvus_token"
   ```

4. Run the application:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8003
   ```

### Docker Deployment

1. Build and run with Docker Compose:
   ```
   docker-compose up -d
   ```

2. The API will be available at `http://localhost:8003`

### Railway Deployment

1. Connect your repository to Railway
2. Set up the following environment variables in Railway:
   - `MILVUS_URI`
   - `MILVUS_TOKEN`
   - `CHROMA_PATH` (if using ChromaDB)
3. Create a volume and mount it to `/app/data` for persistent storage
4. Deploy your application

## API Endpoints

### Collections

- `POST /collections` - Create a new collection
- `GET /collections` - List all collections
- `GET /collections/{collection_name}` - Get collection details
- `DELETE /collections/{collection_name}` - Delete a collection

### Documents

- `POST /collections/{collection_name}/add` - Add documents to a collection
- `POST /collections/{collection_name}/query` - Query a collection
- `GET /collections/{collection_name}/peek` - Peek at documents in a collection

## Usage Examples

### Creating a Collection

```python
import requests

# Create a ChromaDB collection
response = requests.post(
    "http://localhost:8003/collections",
    json={"name": "my_collection", "vector_store": "chroma"}
)

# Create a Milvus collection
response = requests.post(
    "http://localhost:8003/collections",
    json={"name": "my_collection", "vector_store": "milvus"}
)
```

### Adding Documents

```python
import requests

documents = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog runs in the park",
    "The fox is quick and brown"
]

response = requests.post(
    "http://localhost:8003/collections/my_collection/add",
    json={
        "documents": documents,
        "vector_store": "chroma"  # or "milvus"
    }
)
```

### Querying Documents

```python
import requests

response = requests.post(
    "http://localhost:8003/collections/my_collection/query",
    json={
        "query_texts": ["quick brown fox"],
        "n_results": 5,
        "vector_store": "chroma",  # or "milvus"
        "rerank": True
    }
)

results = response.json()
for result in results:
    print(f"Text: {result['text']}")
    print(f"Score: {result['score']}")
    print(f"Reranked: {result.get('reranked', False)}")
    print("---")
```

## Performance Optimizations

The vector store implementation includes several optimizations:

- **Embedding Caching**: Avoids regenerating embeddings for the same text
- **Reranking Caching**: Caches reranking results for the same query and texts
- **Collection Loading Cache**: Tracks loaded collections to avoid repeated loading/unloading
- **Batch Processing**: Processes documents in batches for better performance
- **Optimized Index Type**: Uses `IVF_SQ8` for Milvus for better speed with minimal accuracy loss
