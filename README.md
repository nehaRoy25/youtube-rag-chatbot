# YouTube Video RAG Chatbot (Python + Streamlit + FAISS + Groq LLM)

A complete, modular **Retrieval-Augmented Generation (RAG)** web application built with **Streamlit**, **LangChain**, **HuggingFace Sentence Transformers**, **FAISS**, and the **Groq API**. Paste any YouTube video URL to extract its transcript, inspect timestamped chunks and vector embeddings, run real semantic similarity search, and chat with a strictly grounded AI assistant that cites chunk IDs and clickable YouTube timestamps.

---

## Architecture Overview

```
YouTube URL
   │
   ▼
[1. URL & Metadata]      modules/utils.py        (Validate URL, extract video ID, title, thumbnail)
   │
   ▼
[2. Transcript]          modules/transcript.py   (youtube-transcript-api extraction + segment offsets)
   │
   ▼
[3. Chunking]            modules/chunking.py     (LangChain RecursiveCharacterTextSplitter + timestamps)
   │
   ▼
[4. Embeddings]          modules/embeddings.py   (SentenceTransformers all-MiniLM-L6-v2 L2-normalized vectors)
   │
   ▼
[5. Vector Store]        modules/vector_store.py (FAISS IndexFlatIP cosine similarity + data/ persistence)
   │
   ▼
[6. Retrieval]           modules/retrieval.py    (Embed user query -> top-K FAISS search + scores)
   │
   ▼
[7. Grounded LLM]        modules/llm.py          (Groq API strict context-only QA + citations)
```

---

## Key Features

- **Video Input & Metadata Card**: Validates standard, shortened (`youtu.be`), Shorts, Live, and Embed YouTube URLs and displays the video thumbnail, title, channel, and video ID.
- **Real Transcript Extraction**: Fetches manual or auto-generated transcripts via `youtube-transcript-api`, handles unavailable/disabled/age-restricted captions cleanly, shows total word count, and provides both readable full text and timestamped line views.
- **Interactive Chunk Explorer**: Splits transcripts using `RecursiveCharacterTextSplitter` with configurable chunk size and overlap, maps exact start/end timestamps to every chunk, lets you inspect individual chunks by ID, and displays all generated chunks with keyword filtering.
- **Embeddings & FAISS Vector Store**: Generates real dense vector representations using HuggingFace `SentenceTransformer` (`all-MiniLM-L6-v2`), displays dimensions and a sample vector preview, stores vectors + metadata in a FAISS `IndexFlatIP` cosine similarity index, and persists indexes under `data/<video_id>/`.
- **Live Similarity Search Inspector**: Displays the query, top 3–5 retrieved chunks, cosine similarity scores with visual progress bars, clickable timestamps, and highlighted matching transcript terms.
- **Strictly Grounded Groq Chatbot**: Answers questions exclusively from the retrieved transcript chunks, cites chunk IDs and clickable YouTube timestamps (`&t=...s`), maintains per-video session chat history, supports clearing chat or replacing the video index, and never exposes the Groq API key in the UI.

---

## Project Structure

```text
youtube-rag-chatbot/
├── app.py                  # Streamlit dashboard & end-to-end RAG workflow
├── config.py               # Environment loading and application defaults
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── README.md               # Documentation & setup guide
├── .gitignore              # Keeps secrets and cached indexes out of git
├── .streamlit/
│   └── secrets.toml.example # Streamlit Secrets template
├── modules/
│   ├── __init__.py
│   ├── utils.py            # URL validation, metadata lookup, timestamps, highlighting
│   ├── transcript.py       # YouTube transcript extraction & normalization
│   ├── chunking.py         # LangChain text splitting & timestamp mapping
│   ├── embeddings.py       # SentenceTransformer encoding & vector summaries
│   ├── vector_store.py     # FAISS IndexFlatIP storage, search & persistence
│   ├── retrieval.py        # Query embedding & top-K similarity retrieval
│   └── llm.py              # Groq LLM client with strict RAG grounding rules
├── assets/
│   └── style.css           # Custom responsive dashboard CSS
└── data/                   # Persisted FAISS indexes and chunk metadata
```

---

## Setup & Installation

### 1. Navigate to the project directory

```bash
cd youtube-rag-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your Groq API Key with Streamlit Secrets

#### Local Development
Create `.streamlit/secrets.toml` from the example template:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and set your Groq API key (from [Groq Console](https://console.groq.com/keys)):

```toml
GROQ_API_KEY = "gsk_your_actual_groq_api_key"
```

#### Hosting on Streamlit Community Cloud
1. Push your repository to GitHub (`.gitignore` automatically excludes `.streamlit/secrets.toml` and `.env`).
2. Deploy `app.py` on [share.streamlit.io](https://share.streamlit.io).
3. In **App Settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_groq_api_key"
   ```

> **Security Note:** Secrets are read server-side via `st.secrets` (with optional environment fallback) and are never displayed or exposed in the Streamlit frontend.

---

## Running the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

Then open the local URL printed in your terminal (typically `http://localhost:8501`).

### Using the App

1. Paste any YouTube video URL (with captions/transcript enabled) into **Section A: Video Input & Processing**.
2. Click **Process Video** and watch live progress through transcript extraction, chunking, embedding generation, and FAISS indexing.
3. Inspect the extracted transcript (**Section B**), generated chunks (**Section C**), and embedding/FAISS metrics (**Section D**).
4. Ask questions in **Section F: AI Chat Interface**.
5. Inspect the top retrieved transcript chunks, cosine similarity scores, and highlighted passages in **Section E: Similarity Search**, and click any timestamp pill to jump directly to that moment on YouTube.
