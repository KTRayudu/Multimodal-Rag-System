# Multimodal RAG System

A Retrieval-Augmented Generation (RAG) prototype that ingests multimodal
files (PDF, text, images via OCR, audio via transcription) into a local
Qdrant vector store and answers questions over them using Google's Gemini
models, behind a Streamlit UI.

## Key Features

-   **Multimodal ingestion:** `.pdf`, `.txt`, images (`.jpg`, `.jpeg`, `.png`)
    via Tesseract OCR, and `.wav` audio via speech-to-text.
-   **Local vector store:** Qdrant in embedded/local-disk mode — no Docker or
    separate server required. Data persists in `./local_qdrant_db`.
-   **Semantic retrieval:** Chunks are embedded with `gemini-embedding-001`
    (768 dimensions) and retrieved by cosine similarity.
-   **Grounded answers:** Gemini answers strictly from retrieved context and
    says so when the context is insufficient.
-   **Interactive UI:** Streamlit app for uploading files and chatting, with
    an expander to inspect the retrieved context behind every answer.
-   **Evaluation suite:** DeepEval metrics (answer relevancy, faithfulness,
    contextual recall) runnable under pytest.

## Architecture

```
upload ─► extract (pypdf / Tesseract / SpeechRecognition)
       ─► chunk (RecursiveCharacterTextSplitter, 1000/100)
       ─► embed (gemini-embedding-001, 768-d)
       ─► store (local Qdrant, cosine)

query  ─► embed ─► top-k semantic search ─► Gemini ─► grounded answer
```

| File | Role |
| --- | --- |
| `app.py` | Streamlit UI (upload + chat) |
| `ingestion.py` | File-type routing, text extraction, chunking |
| `utils.py` | Model config, embeddings, OCR, transcription, chunking |
| `vector_db.py` | Qdrant client, collection setup, upsert, search |
| `hybrid_search.py` | Retrieval + answer generation |
| `evaluation.py` | DeepEval/pytest quality suite |
| `knowledge_graph.py` | Optional Neo4j helpers — **not used** by `app.py` |

> **Note:** an earlier version combined graph traversal (Neo4j) with semantic
> search. The graph was removed from the live retrieval path; `app.py` is
> Qdrant-only. `knowledge_graph.py` remains for reference.

## Tech Stack

-   **Backend:** Python 3.10+, LangChain, Streamlit
-   **Vector DB:** Qdrant (embedded, local disk)
-   **LLM & Embeddings:** Google Gemini (`gemini-2.5-flash-lite`, `gemini-embedding-001`)
-   **Evaluation:** DeepEval + pytest

---

## Prerequisites

1.  **Python 3.10+** — [Download Python](https://www.python.org/downloads/)
2.  **Tesseract OCR engine** — only needed for image ingestion.
    -   **macOS:** `brew install tesseract`
    -   **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
    -   **Windows:** [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
3.  **Google AI API key** — free tier is sufficient.
    Get one from [Google AI Studio](https://aistudio.google.com/apikey).

Docker is **not** required: Qdrant runs embedded, writing to `./local_qdrant_db`.

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

-   **macOS / Linux:** `source .venv/bin/activate`
-   **Windows:** `.\.venv\Scripts\activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Copy the example file and add your key:

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Then edit `.env`:

```dotenv
GOOGLE_API_KEY=your_api_key_here
```

`.env` is gitignored — never commit your key.

---

## Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

1.  Upload one or more files in the sidebar and click **Process Files**.
2.  Ask questions in the chat box.
3.  Expand **View Retrieved Context** to see the chunks behind each answer.

### Notes and limitations

-   **Audio:** only `.wav` is supported. `SpeechRecognition` cannot decode
    MP3 — convert first: `ffmpeg -i input.mp3 output.wav`.
-   **Rate limits:** chunks are embedded in batches of 10 with a 2s pause and
    automatic retry/backoff on HTTP 429, to stay within free-tier quota.
-   **Changing the embedding model:** `EMBEDDING_MODEL` and `EMBEDDING_DIM` in
    `utils.py` must stay in sync with the stored vectors. Changing either one
    makes existing vectors incomparable — delete `local_qdrant_db/` and
    re-ingest your files.
-   **Single writer:** embedded Qdrant locks its directory, so only one
    process may use `local_qdrant_db` at a time. Stop the Streamlit app before
    running scripts against the same store.

---

## Running the Evaluation Suite

The DeepEval judge metrics default to OpenAI models, so scoring needs an
OpenAI key **in addition to** the Google key used by the pipeline:

```bash
# in .env
OPENAI_API_KEY=your_openai_key
DEEPEVAL_JUDGE_MODEL=gpt-4o-mini   # optional, this is the default
```

Then:

```bash
pytest evaluation.py -v
```

The suite ingests a small fixed document into a temporary Qdrant store (your
real `local_qdrant_db` is untouched) and scores four queries for answer
relevancy, faithfulness, and contextual recall.
