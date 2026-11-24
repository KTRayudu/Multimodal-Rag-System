import os
import time
from qdrant_client import QdrantClient, models
import utils

def get_qdrant_client():
    """Initializes Qdrant in 'Local Disk Mode'."""
    client = QdrantClient(path="./local_qdrant_db")
    return client

def create_collection_if_not_exists(client: QdrantClient, collection_name: str):
    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if collection_name not in collection_names:
            print(f"Creating collection: {collection_name}")
            # We use 768 because 'text-embedding-004' is 768 dimensions
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            )
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        print(f"An error occurred while creating collection: {e}")

def upsert_chunks(client: QdrantClient, collection_name: str, chunks: list[str], metadatas: list[dict]):
    """
    Embeds text in very small batches with auto-retry for 429 errors.
    """
    if not chunks:
        return

    print(f"Processing {len(chunks)} chunks...")
    embedding_model = utils.get_embedding_model()
    
    # --- HEAVY BATCHING LOGIC ---
    BATCH_SIZE = 10  # Reduced to 10 to survive Free Tier
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_chunks)
        batch_chunks = chunks[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]
        
        print(f"Embedding batch {i} to {batch_end}...")
        
        retries = 3
        while retries > 0:
            try:
                # 1. Embed the current batch
                vectors = embedding_model.embed_documents(batch_chunks)

                points = [
                    models.PointStruct(
                        id=k + i, 
                        vector=vector,
                        payload={"text": chunk, **metadata}
                    )
                    for k, (vector, chunk, metadata) in enumerate(zip(vectors, batch_chunks, batch_metadatas))
                ]

                # 2. Upload batch to Qdrant
                client.upsert(collection_name=collection_name, points=points)
                
                # Success! Break the retry loop and move to next batch
                time.sleep(2) # Short rest between successful batches
                break 

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    print(f"⚠️ Hit Rate Limit (429). Waiting 60 seconds for quota to reset...")
                    time.sleep(60) # WAIT 1 MINUTE
                    retries -= 1
                else:
                    print(f"❌ Error on batch {i}-{batch_end}: {e}")
                    break # Unknown error, stop retrying this batch

    print(f"Successfully processed all chunks.")

def semantic_search(client: QdrantClient, collection_name: str, query_text: str, limit: int = 5):
    if not query_text:
        return []
        
    embedding_model = utils.get_embedding_model()
    
    try:
        query_vector = embedding_model.embed_query(query_text)

        # Using query_points which is more robust for local mode
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        ).points
        
        return search_results
    except Exception as e:
        print(f"An error occurred during semantic search: {e}")
        return []