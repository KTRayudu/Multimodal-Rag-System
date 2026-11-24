import os
import streamlit as st
import utils
import vector_db
import ingestion
import hybrid_search

st.set_page_config(page_title="Local RAG with Gemini", page_icon="💻", layout="wide")

QDRANT_COLLECTION_NAME = "local_rag_collection"
UPLOAD_DIR = "temp_uploads"

utils.load_environment()




if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@st.cache_resource
def get_db_connection():
    """
    Initializes ONLY the Local Qdrant connection.
    """
    print("Initializing Local Qdrant...")
    qdrant_client = vector_db.get_qdrant_client()
    vector_db.create_collection_if_not_exists(qdrant_client, QDRANT_COLLECTION_NAME)
    return qdrant_client

try:
    qdrant_client = get_db_connection()
    st.success("Connected to Local Qdrant Database.")
except Exception as e:
    st.error(f"Failed to connect to database: {e}")
    st.stop()


st.title("💻 Local RAG System (Gemini Powered)")
st.markdown("Files are saved to your hard drive. AI runs via Gemini API.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Upload Knowledge")
    uploaded_files = st.file_uploader(
        "Upload Files", 
        type=['pdf', 'txt', 'jpg', 'png', 'mp3'],
        accept_multiple_files=True
    )

    if st.button("Process Files"):
        if uploaded_files:
            with st.spinner("Ingesting..."):
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        # UPDATED CALL: Removed neo4j_driver argument
                        ingestion.ingest_file(
                            file_path=temp_path,
                            collection_name=QDRANT_COLLECTION_NAME,
                            qdrant_client=qdrant_client
                        )
                        st.success(f"Ingested: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        os.remove(temp_path)
        else:
            st.warning("Upload a file first.")

# --- CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # UPDATED CALL: Removed neo4j_driver argument
            context = hybrid_search.hybrid_retrieval(
                prompt,
                QDRANT_COLLECTION_NAME,
                qdrant_client
            )
            
            final_answer = hybrid_search.generate_response(prompt, context)
            
            st.markdown(final_answer)
            
            with st.expander("View Retrieved Context"):
                st.text(context)
                
    st.session_state.messages.append({"role": "assistant", "content": final_answer})