import utils
import vector_db

def hybrid_retrieval(user_query: str, collection_name: str, qdrant_client) -> str:
    """
    Performs Semantic Search only (since we removed the Graph).
    """
    print("\n--- Starting Retrieval ---")
    
    # 1. Semantic Search from Local Vector DB
    print("Performing semantic search...")
    semantic_results = vector_db.semantic_search(
        client=qdrant_client,
        collection_name=collection_name,
        query_text=user_query,
        limit=5 
    )
    
    # Combine results into a single context string
    context = "\n".join([hit.payload['text'] for hit in semantic_results])
    print(f"Found {len(semantic_results)} chunks.")
    
    return context


def generate_response(query: str, context: str) -> str:
    """
    Generates the final answer using Gemini.
    """
    llm = utils.get_llm()
    prompt = f"""
    You are an intelligent assistant. Answer the user's question based *only* on the provided context. 
    If the answer is not in the context, say "I don't have enough information."

    USER QUESTION:
    {query}

    CONTEXT:
    {context}

    ANSWER:
    """
    
    try:
        answer = llm.invoke(prompt)
        return answer.content.strip()
    except Exception as e:
        return f"Error generating response: {e}"