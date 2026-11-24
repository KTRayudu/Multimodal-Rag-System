import os
import pypdf
import utils
import vector_db

# --- Helper Functions (Same as original) ---
def _extract_text_from_pdf(file_path: str) -> str:
    try:
        reader = pypdf.PdfReader(file_path)
        text = "".join(page.extract_text() for page in reader.pages)
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def _extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return f.read()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return ""

def _extract_text_from_image(file_path: str) -> str:
    return utils.ocr_from_image(file_path)

def _transcribe_audio(file_path: str) -> str:
    return utils.transcribe_audio(file_path)

# --- Main Ingestion Logic ---

def ingest_file(file_path: str, collection_name: str, qdrant_client):
    """
    Ingests a file into Local Qdrant ONLY. (No Neo4j)
    """
    filename = os.path.basename(file_path)
    print(f"\n--- Starting ingestion for: {filename} ---")

    # 1. Determine file type & extract text
    file_extension = os.path.splitext(filename)[1].lower()
    raw_text = ""
    
    if file_extension == '.pdf':
        raw_text = _extract_text_from_pdf(file_path)
    elif file_extension == '.txt':
        raw_text = _extract_text_from_txt(file_path)
    elif file_extension in ['.jpg', '.jpeg', '.png']:
        raw_text = _extract_text_from_image(file_path)
    elif file_extension in ['.mp3', '.wav']:
        raw_text = _transcribe_audio(file_path)
    else:
        print(f"Unsupported file type: {file_extension}")
        return

    if not raw_text:
        print(f"No text extracted from {filename}.")
        return

    # 2. Chunk the text
    text_chunks = utils.chunk_text(raw_text)
    print(f"Text split into {len(text_chunks)} chunks.")

    # 3. Upsert to Local Qdrant
    metadatas = [{"source_file": filename, "chunk_index": i} for i in range(len(text_chunks))]
    vector_db.upsert_chunks(qdrant_client, collection_name, text_chunks, metadatas)

    print(f"--- Finished ingestion for: {filename} ---")