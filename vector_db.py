import time
import uuid

from qdrant_client import QdrantClient, models

import utils


def get_qdrant_client():
    """Initializes Qdrant in 'Local Disk Mode'."""
    return QdrantClient(path="./local_qdrant_db")


def create_collection_if_not_exists(client: QdrantClient, collection_name: str):
    """
    Creates the collection only when it is missing.

    Uses create_collection (not recreate_collection) so that an existing
    collection is never dropped.
    """
    try:
        existing = {c.name for c in client.get_collections().collections}

        if collection_name in existing:
            print(f"Collection '{collection_name}' already exists.")
            return

        print(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=utils.EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        print(f"An error occurred while creating collection: {e}")
        raise


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[str],
    metadatas: list[dict],
):
    """
    Embeds text in small batches and upserts it into Qdrant.

    Retries on 429 (quota) errors. Raises RuntimeError if nothing could be
    stored, so callers do not report a successful ingestion that did not
    actually happen.
    """
    if not chunks:
        return

    print(f"Processing {len(chunks)} chunks...")

    BATCH_SIZE = 10
    total_chunks = len(chunks)
    stored = 0
    failed = 0

    for i in range(0, total_chunks, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_chunks)
        batch_chunks = chunks[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]

        print(f"Embedding batch {i} to {batch_end}...")

        retries = 3
        while retries > 0:
            try:
                vectors = utils.embed_documents(batch_chunks)

                points = [
                    models.PointStruct(
                        # Random UUIDs: sequential per-file ids collided
                        # across files and silently overwrote earlier points.
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={"text": chunk, **metadata},
                    )
                    for vector, chunk, metadata in zip(
                        vectors, batch_chunks, batch_metadatas
                    )
                ]

                client.upsert(collection_name=collection_name, points=points)
                stored += len(points)

                time.sleep(2)  # Short rest between successful batches
                break

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    retries -= 1
                    if retries == 0:
                        print(
                            f"[ERROR] Rate limit persisted, giving up on "
                            f"batch {i}-{batch_end}."
                        )
                        failed += len(batch_chunks)
                        break
                    print("[WARN] Hit rate limit (429). Waiting 60s for quota...")
                    time.sleep(60)
                else:
                    print(f"[ERROR] Batch {i}-{batch_end} failed: {e}")
                    failed += len(batch_chunks)
                    break

    print(f"Stored {stored}/{total_chunks} chunks ({failed} failed).")

    if stored == 0:
        raise RuntimeError(
            "No chunks could be embedded or stored. See the errors above."
        )


def semantic_search(
    client: QdrantClient,
    collection_name: str,
    query_text: str,
    limit: int = 5,
):
    if not query_text:
        return []

    try:
        query_vector = utils.embed_query(query_text)

        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        ).points

        return search_results
    except Exception as e:
        print(f"An error occurred during semantic search: {e}")
        return []
