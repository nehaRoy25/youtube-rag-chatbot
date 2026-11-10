"""Streamlit frontend and RAG pipeline orchestrator for YouTube Video Q&A."""

import html
from typing import Optional

import streamlit as st

import config
from modules.chunking import ChunkingStats, TextChunk, chunk_transcript
from modules.embeddings import EmbeddingError, EmbeddingManager, EmbeddingSummary
from modules.llm import GroqAnswerGenerator, LLMError, fetch_available_groq_models
from modules.retrieval import RetrievalError, RetrievalResult, Retriever
from modules.transcript import (
    TranscriptResult,
    TranscriptUnavailableError,
    extract_transcript,
)
from modules.utils import (
    InvalidYouTubeURLError,
    VideoMetadata,
    extract_video_id,
    fetch_video_metadata,
    highlight_query_terms,
)
from modules.vector_store import FAISSVectorStore, VectorStoreError

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_custom_css() -> None:
    """Inject custom CSS from assets/style.css."""
    if config.STYLE_PATH.exists():
        css = config.STYLE_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _init_session_state() -> None:
    """Initialize all Streamlit session state variables once."""
    defaults = {
        "video_url_input": "",
        "video_metadata": None,
        "transcript": None,
        "chunks": [],
        "chunk_stats": None,
        "embedding_summary": None,
        "embedding_manager": None,
        "vector_store": None,
        "last_retrieval": None,
        "chat_history": [],
        "pipeline_stage": "idle",
        "processing_error": None,
        "indexed_signature": None,
        "pending_prompt": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_session_pipeline(clear_url: bool = False) -> None:
    """Clear current vector store, transcript, chunks, and chat history."""
    existing_store: Optional[FAISSVectorStore] = st.session_state.get("vector_store")
    if existing_store is not None:
        existing_store.clear()

    if clear_url:
        st.session_state.video_url_input = ""
    st.session_state.video_metadata = None
    st.session_state.transcript = None
    st.session_state.chunks = []
    st.session_state.chunk_stats = None
    st.session_state.embedding_summary = None
    st.session_state.vector_store = None
    st.session_state.last_retrieval = None
    st.session_state.chat_history = []
    st.session_state.pipeline_stage = "idle"
    st.session_state.processing_error = None
    st.session_state.indexed_signature = None
    st.session_state.pending_prompt = None


def _is_pipeline_ready() -> bool:
    """Return True when a processed video is fully indexed and ready for questions."""
    store: Optional[FAISSVectorStore] = st.session_state.get("vector_store")
    emb_mgr: Optional[EmbeddingManager] = st.session_state.get("embedding_manager")
    return bool(
        st.session_state.get("pipeline_stage") == "ready"
        and store is not None
        and store.is_ready
        and emb_mgr is not None
    )


def _render_sidebar() -> dict:
    """Render sidebar configuration controls and pipeline status checklist."""
    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Configuration")

        # Secure API key status check (never displays or leaks the key itself)
        if config.is_groq_configured():
            secret_source = html.escape(config.get_secret_source())
            st.markdown(
                f'<span class="status-badge badge-ready">● Groq API Ready ({secret_source})</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge badge-warning">● Groq API Key Missing</span>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Add `GROQ_API_KEY` to **Streamlit Secrets** (`.streamlit/secrets.toml` locally "
                "or **App Settings → Secrets** on Streamlit Cloud)."
            )

        st.divider()

        st.markdown("#### 🤖 Groq LLM Settings")
        groq_models = fetch_available_groq_models()
        preferred_groq_model = config.get_groq_model()
        default_model_idx = (
            groq_models.index(preferred_groq_model)
            if preferred_groq_model in groq_models
            else 0
        )
        groq_model = st.selectbox(
            "Groq Model",
            options=groq_models,
            index=default_model_idx,
            help="Active Groq-hosted chat LLM used for strictly grounded answers.",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=0.5,
            value=config.DEFAULT_TEMPERATURE,
            step=0.05,
            help="Lower values keep answers strictly factual and deterministic.",
        )

        st.markdown("#### 🧬 Embeddings & Chunking")
        preferred_emb_model = config.get_embedding_model()
        default_emb_idx = (
            config.AVAILABLE_EMBEDDING_MODELS.index(preferred_emb_model)
            if preferred_emb_model in config.AVAILABLE_EMBEDDING_MODELS
            else 0
        )
        embedding_model = st.selectbox(
            "Sentence Transformer Model",
            options=config.AVAILABLE_EMBEDDING_MODELS,
            index=default_emb_idx,
            help="HuggingFace embedding model used for chunks and user questions.",
        )
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=config.MIN_CHUNK_SIZE,
            max_value=config.MAX_CHUNK_SIZE,
            value=config.DEFAULT_CHUNK_SIZE,
            step=50,
        )
        max_allowed_overlap = min(config.MAX_CHUNK_OVERLAP, chunk_size - 50)
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=config.MIN_CHUNK_OVERLAP,
            max_value=max(0, max_allowed_overlap),
            value=min(config.DEFAULT_CHUNK_OVERLAP, max(0, max_allowed_overlap)),
            step=25,
        )

        st.markdown("#### 🔍 Retrieval Settings")
        top_k = st.slider(
            "Top-K Retrieved Chunks",
            min_value=config.MIN_TOP_K,
            max_value=config.MAX_TOP_K,
            value=config.DEFAULT_TOP_K,
            step=1,
            help="Number of most similar chunks retrieved per question (3–5).",
        )

        # Warn if settings changed after indexing
        current_sig = st.session_state.get("indexed_signature")
        selected_sig = (embedding_model, chunk_size, chunk_overlap)
        if _is_pipeline_ready() and current_sig and current_sig != selected_sig:
            st.info(
                "Chunking or embedding settings changed. Click **Process Video** to re-index."
            )

        st.divider()
        st.markdown("#### 📊 Workflow Status")
        has_meta = st.session_state.video_metadata is not None
        has_transcript = st.session_state.transcript is not None
        has_chunks = len(st.session_state.chunks) > 0
        has_vectors = _is_pipeline_ready()

        steps = [
            ("1. Video Metadata", has_meta),
            ("2. Transcript Extracted", has_transcript),
            ("3. Chunks Created", has_chunks),
            ("4. Embeddings & FAISS", has_vectors),
        ]
        for label, done in steps:
            badge = (
                '<span class="status-badge badge-ready">Done</span>'
                if done
                else '<span class="status-badge badge-info">Pending</span>'
            )
            st.markdown(
                f'<div class="sidebar-step"><span>{html.escape(label)}</span>{badge}</div>',
                unsafe_allow_html=True,
            )

        st.write("")
        if st.button("🔄 Reset / New Video", width="stretch"):
            _reset_session_pipeline(clear_url=True)
            st.rerun()

    return {
        "groq_model": groq_model,
        "temperature": temperature,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
    }


def _process_video_pipeline(
    url: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Execute the full RAG ingestion pipeline with live progress updates."""
    st.session_state.processing_error = None

    with st.status("Processing YouTube video through RAG pipeline...", expanded=True) as status_box:
        progress_bar = st.progress(0, text="Validating YouTube URL...")

        try:
            # Step 1: Validate URL & fetch metadata
            video_id = extract_video_id(url)
            st.write("🔎 Fetching video title, channel, and thumbnail...")
            progress_bar.progress(10, text="Fetching video metadata...")
            metadata = fetch_video_metadata(video_id, original_url=url)

            # Step 2: Extract transcript
            st.write("📜 Extracting spoken transcript from YouTube...")
            progress_bar.progress(25, text="Extracting YouTube transcript...")
            transcript = extract_transcript(video_id)
            st.write(
                f"✅ Extracted **{transcript.word_count:,} words** "
                f"({len(transcript.segments)} segments, {transcript.language})."
            )

            # Step 3: Text chunking
            st.write(
                f"🧩 Splitting transcript into chunks (`size={chunk_size}`, `overlap={chunk_overlap}`)..."
            )
            progress_bar.progress(45, text="Chunking transcript text...")
            chunks, chunk_stats = chunk_transcript(
                transcript=transcript,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.write(f"✅ Generated **{chunk_stats.total_chunks} chunks** with mapped timestamps.")

            # Step 4: Generate embeddings
            st.write(f"🧬 Loading embedding model `{embedding_model}` & generating vectors...")
            progress_bar.progress(55, text=f"Loading {embedding_model}...")
            embedding_manager = EmbeddingManager(model_name=embedding_model)

            def _on_embed_progress(done: int, total: int) -> None:
                pct = 55 + int((done / max(1, total)) * 30)
                progress_bar.progress(pct, text=f"Generating embeddings ({done}/{total} chunks)...")

            chunk_texts = [c.content for c in chunks]
            embeddings = embedding_manager.embed_texts(
                chunk_texts,
                batch_size=32,
                progress_callback=_on_embed_progress,
            )
            embedding_summary = embedding_manager.build_summary(embeddings)
            st.write(
                f"✅ Created **{embedding_summary.total_embeddings} vectors** "
                f"({embedding_summary.dimension} dimensions)."
            )

            # Step 5: Store in FAISS & replace any previous store
            st.write("🗄️ Indexing vectors & metadata in FAISS vector database...")
            progress_bar.progress(90, text="Building FAISS index...")
            old_store: Optional[FAISSVectorStore] = st.session_state.get("vector_store")
            if old_store is not None:
                old_store.clear()

            new_store = FAISSVectorStore(
                dimension=embedding_manager.dimension,
                video_id=video_id,
                embedding_model=embedding_model,
            )
            new_store.add_chunks(chunks, embeddings)
            try:
                new_store.save(config.DATA_DIR)
            except OSError:
                pass

            # Persist new pipeline state & reset session chat for new video
            st.session_state.video_metadata = metadata
            st.session_state.transcript = transcript
            st.session_state.chunks = chunks
            st.session_state.chunk_stats = chunk_stats
            st.session_state.embedding_manager = embedding_manager
            st.session_state.embedding_summary = embedding_summary
            st.session_state.vector_store = new_store
            st.session_state.last_retrieval = None
            st.session_state.chat_history = []
            st.session_state.pipeline_stage = "ready"
            st.session_state.indexed_signature = (embedding_model, chunk_size, chunk_overlap)

            progress_bar.progress(100, text="Pipeline ready! Ask any question below.")
            status_box.update(
                label="✅ Video processed and indexed in FAISS! Ready for Q&A.",
                state="complete",
                expanded=False,
            )

        except (
            InvalidYouTubeURLError,
            TranscriptUnavailableError,
            EmbeddingError,
            VectorStoreError,
            ValueError,
        ) as exc:
            st.session_state.processing_error = str(exc)
            progress_bar.empty()
            status_box.update(label="❌ Video processing failed", state="error", expanded=True)
            st.error(str(exc))
        except Exception as exc:
            st.session_state.processing_error = f"Unexpected processing error: {exc}"
            progress_bar.empty()
            status_box.update(label="❌ Video processing failed", state="error", expanded=True)
            st.error(st.session_state.processing_error)


def _handle_user_question(question: str, settings: dict) -> None:
    """Run query embedding, FAISS similarity search, and Groq answer generation."""
    if not _is_pipeline_ready():
        st.warning("Please process a YouTube video before asking questions.")
        return

    emb_mgr: EmbeddingManager = st.session_state.embedding_manager
    vector_store: FAISSVectorStore = st.session_state.vector_store
    metadata: VideoMetadata = st.session_state.video_metadata

    retriever = Retriever(embedding_manager=emb_mgr, vector_store=vector_store)

    try:
        retrieval_result = retriever.retrieve(question, top_k=settings["top_k"])
        st.session_state.last_retrieval = retrieval_result
    except (RetrievalError, EmbeddingError, VectorStoreError) as exc:
        st.error(f"Similarity search failed: {exc}")
        return

    prior_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.chat_history
    ]

    st.session_state.chat_history.append({"role": "user", "content": question.strip()})

    generator = GroqAnswerGenerator(
        model=settings["groq_model"],
        temperature=settings["temperature"],
    )

    try:
        llm_response = generator.generate_answer(
            question=question,
            retrieval_result=retrieval_result,
            video_title=metadata.title if metadata else "",
            chat_history=prior_history,
        )
        sources = [
            {
                "rank": item.rank,
                "chunk_id": item.chunk.chunk_id,
                "timestamp": item.chunk.timestamp,
                "timestamp_range": item.chunk.timestamp_range,
                "youtube_url": item.chunk.youtube_url,
                "similarity_score": item.similarity_score,
                "similarity_pct": item.similarity_pct,
            }
            for item in retrieval_result.results
        ]
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": llm_response.answer,
                "sources": sources,
                "model": llm_response.model,
                "tokens": llm_response.total_tokens,
            }
        )
    except LLMError as exc:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"⚠️ Could not generate answer: {exc}",
                "sources": [],
                "model": settings["groq_model"],
                "tokens": 0,
            }
        )


def _render_hero_header() -> None:
    """Render the top hero banner."""
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-top-row">
                <span class="hero-pill">▶ YouTube RAG Assistant</span>
                <span class="hero-stack-pill">LangChain • SentenceTransformers • FAISS • Groq LLM</span>
            </div>
            <h1 class="hero-title">{html.escape(config.APP_TITLE)}</h1>
            <p class="hero-subtitle">{html.escape(config.APP_SUBTITLE)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_video_input_section(settings: dict) -> None:
    """Render Section A: YouTube URL Input, Process Video button, and Video Info Card."""
    st.markdown(
        """
        <div class="section-header">
            <h2 class="section-title">🎬 A. Video Input &amp; Processing</h2>
            <span class="section-tag">Step 1</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_url, col_btn = st.columns([4.2, 1.1], vertical_alignment="bottom")
    with col_url:
        url_input = st.text_input(
            "YouTube Video URL",
            key="video_url_input",
            placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
        )
    with col_btn:
        process_clicked = st.button(
            "Process Video",
            type="primary",
            width="stretch",
        )

    if process_clicked:
        _process_video_pipeline(
            url=url_input,
            embedding_model=settings["embedding_model"],
            chunk_size=settings["chunk_size"],
            chunk_overlap=settings["chunk_overlap"],
        )

    metadata: Optional[VideoMetadata] = st.session_state.get("video_metadata")
    transcript: Optional[TranscriptResult] = st.session_state.get("transcript")

    if metadata is not None:
        st.markdown('<div class="video-info-card">', unsafe_allow_html=True)
        c_thumb, c_info = st.columns([1.25, 2.75], vertical_alignment="center")
        with c_thumb:
            st.image(metadata.thumbnail_url, width="stretch")
        with c_info:
            status_badge = (
                '<span class="status-badge badge-ready">● Indexed &amp; Ready</span>'
                if _is_pipeline_ready()
                else '<span class="status-badge badge-warning">● Pending Index</span>'
            )
            lang_chip = (
                f'<span class="meta-chip">🗣 {html.escape(transcript.language)}</span>'
                if transcript
                else ""
            )
            dur_chip = (
                f'<span class="meta-chip">⏱ Duration: {html.escape(transcript.formatted_duration)}</span>'
                if transcript
                else ""
            )
            st.markdown(
                f"""
                <div class="video-meta-row">
                    {status_badge}
                    <span class="meta-chip">🆔 {html.escape(metadata.video_id)}</span>
                </div>
                <div class="video-meta-title">{html.escape(metadata.title)}</div>
                <div class="video-meta-row">
                    <span class="meta-chip">📺 {html.escape(metadata.channel)}</span>
                    {lang_chip}
                    {dur_chip}
                </div>
                <a class="meta-link" href="{html.escape(metadata.url)}" target="_blank" rel="noopener noreferrer">
                    ▶ Open Original Video on YouTube ↗
                </a>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # KPI Overview Strip
        chunk_stats: Optional[ChunkingStats] = st.session_state.get("chunk_stats")
        emb_summary: Optional[EmbeddingSummary] = st.session_state.get("embedding_summary")
        vector_store: Optional[FAISSVectorStore] = st.session_state.get("vector_store")
        vs_stats = vector_store.get_stats() if vector_store else None

        st.write("")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            wc = f"{transcript.word_count:,}" if transcript else "0"
            segs = f"{len(transcript.segments)} segments" if transcript else "—"
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Transcript Words</div>'
                f'<div class="kpi-value">{wc}</div>'
                f'<div class="kpi-sub">{segs}</div></div>',
                unsafe_allow_html=True,
            )
        with k2:
            tc = f"{chunk_stats.total_chunks}" if chunk_stats else "0"
            avg = f"Avg {chunk_stats.avg_char_count} chars" if chunk_stats else "—"
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Total Chunks</div>'
                f'<div class="kpi-value">{tc}</div>'
                f'<div class="kpi-sub">{avg}</div></div>',
                unsafe_allow_html=True,
            )
        with k3:
            dim = f"{emb_summary.dimension}d" if emb_summary else "—"
            mdl = emb_summary.model_name if emb_summary else "—"
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Embedding Dim</div>'
                f'<div class="kpi-value">{dim}</div>'
                f'<div class="kpi-sub">{html.escape(mdl)}</div></div>',
                unsafe_allow_html=True,
            )
        with k4:
            vecs = f"{vs_stats.total_vectors}" if vs_stats else "0"
            v_status = vs_stats.status if vs_stats else "Empty"
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">FAISS Vectors</div>'
                f'<div class="kpi-value">{vecs}</div>'
                f'<div class="kpi-sub">Status: {html.escape(v_status)}</div></div>',
                unsafe_allow_html=True,
            )

        if (
            transcript is not None
            and not transcript.language_code.startswith("en")
            and emb_summary is not None
            and "multilingual" not in emb_summary.model_name.lower()
        ):
            st.info(
                f"🌐 Non-English transcript detected (**{transcript.language}**). "
                "For best English-to-video semantic search accuracy, select "
                "`paraphrase-multilingual-MiniLM-L12-v2` in the sidebar and click **Process Video**."
            )


def _render_chat_and_similarity_section(settings: dict) -> None:
    """Render Section F (AI Chat Interface) and Section E (Similarity Search Inspector)."""
    ready = _is_pipeline_ready()

    col_chat, col_sim = st.columns([1.45, 1.0], gap="large")

    # Section F: AI Chat Interface
    with col_chat:
        hdr_l, hdr_r = st.columns([3.2, 1.0], vertical_alignment="center")
        with hdr_l:
            st.markdown(
                """
                <div class="section-header">
                    <h2 class="section-title">💬 F. AI Chat Interface</h2>
                    <span class="section-tag">Groq RAG</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with hdr_r:
            if st.button(
                "🗑️ Clear Chat",
                width="stretch",
                disabled=len(st.session_state.chat_history) == 0,
            ):
                st.session_state.chat_history = []
                st.session_state.last_retrieval = None
                st.rerun()

        if not ready:
            st.info(
                "Paste a YouTube video URL above and click **Process Video** to extract "
                "its transcript, build FAISS embeddings, and unlock the chatbot."
            )

        chat_container = st.container(height=470, border=True)
        with chat_container:
            if not st.session_state.chat_history:
                if ready:
                    st.markdown(
                        "👋 **Video indexed and ready!** Ask any question about the video's "
                        "content below. Answers are strictly grounded in the retrieved transcript "
                        "with chunk IDs and clickable YouTube timestamps."
                    )
                    q1, q2, q3 = st.columns(3)
                    if q1.button("📝 Summarize key points", width="stretch"):
                        st.session_state.pending_prompt = "What are the main points covered in this video?"
                    if q2.button("💡 Main takeaways", width="stretch"):
                        st.session_state.pending_prompt = "What are the key takeaways or conclusions?"
                    if q3.button("🔍 Topics discussed", width="stretch"):
                        st.session_state.pending_prompt = "List the main topics discussed with timestamps."
                else:
                    st.caption("Chat will appear here once a YouTube video has been processed.")
            else:
                for msg in st.session_state.chat_history:
                    role = msg["role"]
                    with st.chat_message(role):
                        st.markdown(msg["content"])
                        if role == "assistant" and msg.get("sources"):
                            pills_html = []
                            for src in msg["sources"]:
                                url = html.escape(src["youtube_url"])
                                cid = html.escape(src["chunk_id"])
                                ts = html.escape(src["timestamp"])
                                score = src["similarity_score"]
                                pills_html.append(
                                    f'<a class="timestamp-pill" href="{url}" target="_blank" '
                                    f'rel="noopener noreferrer" title="Open video at {ts}">'
                                    f"▶ {ts} • {cid} ({score:.2f})</a>"
                                )
                            st.markdown(
                                '<div class="citation-strip">'
                                '<span class="citation-label">Sources:</span>'
                                + "".join(pills_html)
                                + "</div>",
                                unsafe_allow_html=True,
                            )

        prompt = st.chat_input(
            placeholder=(
                "Ask a question about this YouTube video..."
                if ready
                else "Process a YouTube video first to enable chat..."
            ),
            disabled=not ready,
        )

        # Handle either typed chat_input or suggested quick-prompt click
        active_question = prompt or st.session_state.pop("pending_prompt", None)
        if active_question and ready:
            with st.spinner("Searching FAISS & generating grounded answer with Groq..."):
                _handle_user_question(active_question, settings)
            st.rerun()

    # Section E: Similarity Search Inspector
    with col_sim:
        st.markdown(
            """
            <div class="section-header">
                <h2 class="section-title">🎯 E. Similarity Search</h2>
                <span class="section-tag">Top-K FAISS</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        retrieval: Optional[RetrievalResult] = st.session_state.get("last_retrieval")
        if retrieval is None:
            st.info(
                "Ask a question in the chat to inspect the query embedding search, "
                "cosine similarity scores, timestamps, and highlighted passages."
            )
        else:
            st.markdown(
                f"**Query:** `{retrieval.query}`  \n"
                f"<small>Retrieved top **{len(retrieval.results)}** chunks "
                f"using **{retrieval.query_dimension}d** cosine similarity</small>",
                unsafe_allow_html=True,
            )

            sim_container = st.container(height=425, border=False)
            with sim_container:
                for item in retrieval.results:
                    c = item.chunk
                    highlighted_body = highlight_query_terms(c.content, retrieval.query)
                    st.markdown(
                        f"""
                        <div class="chunk-card">
                            <div class="chunk-card-header">
                                <div>
                                    <span class="chunk-id-pill">#{item.rank} • {html.escape(c.chunk_id)}</span>
                                    <a class="timestamp-pill" href="{html.escape(c.youtube_url)}" target="_blank" rel="noopener noreferrer">
                                        ▶ {html.escape(c.timestamp_range)}
                                    </a>
                                </div>
                                <span class="score-pill">Score: {item.similarity_score:.4f} ({item.similarity_pct:.1f}%)</span>
                            </div>
                            <div class="similarity-bar-bg">
                                <div class="similarity-bar-fill" style="width: {item.similarity_pct:.1f}%;"></div>
                            </div>
                            <div class="chunk-body">{highlighted_body}</div>
                            <div class="chunk-footer">
                                <span>Size: {c.char_count} chars</span>
                                <span>Words: {c.word_count}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


def _render_transcript_section() -> None:
    """Render Section B: Expandable Complete Transcript with word count and timestamped view."""
    transcript: Optional[TranscriptResult] = st.session_state.get("transcript")

    st.markdown(
        """
        <div class="section-header">
            <h2 class="section-title">📜 B. Transcript Section</h2>
            <span class="section-tag">Step 2</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if transcript is None:
        st.caption("Extracted video transcript and word count will appear here after processing a video.")
        return

    with st.expander(
        f"View Complete Extracted Transcript — {transcript.word_count:,} words "
        f"({transcript.language})",
        expanded=False,
    ):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Word Count", f"{transcript.word_count:,}")
        m2.metric("Characters", f"{transcript.char_count:,}")
        m3.metric("Spoken Segments", f"{len(transcript.segments):,}")
        m4.metric("Caption Type", "Auto-generated" if transcript.is_generated else "Manual")

        view_mode = st.radio(
            "Display Format",
            options=["Readable Full Text", "Timestamped Lines"],
            horizontal=True,
            key="transcript_view_mode",
        )

        if view_mode == "Readable Full Text":
            st.markdown(
                f'<div class="transcript-box">{html.escape(transcript.full_text)}</div>',
                unsafe_allow_html=True,
            )
        else:
            lines_html = []
            for seg in transcript.segments:
                ts_url = f"https://www.youtube.com/watch?v={transcript.video_id}&t={int(seg.start)}s"
                lines_html.append(
                    f'<div style="margin-bottom:0.4rem;">'
                    f'<a class="timestamp-pill" href="{html.escape(ts_url)}" target="_blank" '
                    f'rel="noopener noreferrer">▶ {html.escape(seg.formatted_start)}</a> '
                    f'<span>{html.escape(seg.text)}</span></div>'
                )
            st.markdown(
                f'<div class="transcript-box">{"".join(lines_html)}</div>',
                unsafe_allow_html=True,
            )


def _render_chunk_explorer_section() -> None:
    """Render Section C: Chunk Explorer showing all chunks + single-chunk inspector."""
    chunks: list[TextChunk] = st.session_state.get("chunks", [])
    chunk_stats: Optional[ChunkingStats] = st.session_state.get("chunk_stats")

    st.markdown(
        """
        <div class="section-header">
            <h2 class="section-title">🧩 C. Chunk Explorer</h2>
            <span class="section-tag">Step 3</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not chunks or chunk_stats is None:
        st.caption("LangChain text chunks, chunk IDs, sizes, and mapped timestamps will appear here.")
        return

    with st.expander(
        f"Inspect All {chunk_stats.total_chunks} Generated Chunks "
        f"(size={chunk_stats.chunk_size_config}, overlap={chunk_stats.chunk_overlap_config})",
        expanded=False,
    ):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Chunks", f"{chunk_stats.total_chunks}")
        c2.metric("Average Size", f"{chunk_stats.avg_char_count} chars")
        c3.metric("Min Chunk Size", f"{chunk_stats.min_char_count} chars")
        c4.metric("Max Chunk Size", f"{chunk_stats.max_char_count} chars")

        st.markdown("#### 🔎 Individual Chunk Inspector")
        chunk_options = {
            f"{c.chunk_id}  |  {c.timestamp_range}  |  {c.char_count} chars": c
            for c in chunks
        }
        selected_label = st.selectbox(
            "Select a Chunk ID to inspect in detail",
            options=list(chunk_options.keys()),
        )
        selected_chunk = chunk_options[selected_label]
        st.markdown(
            f"""
            <div class="chunk-card">
                <div class="chunk-card-header">
                    <span class="chunk-id-pill">{html.escape(selected_chunk.chunk_id)}</span>
                    <a class="timestamp-pill" href="{html.escape(selected_chunk.youtube_url)}" target="_blank" rel="noopener noreferrer">
                        ▶ Jump to {html.escape(selected_chunk.timestamp_range)}
                    </a>
                </div>
                <div class="chunk-body">{html.escape(selected_chunk.content)}</div>
                <div class="chunk-footer">
                    <span>Chunk Size: {selected_chunk.char_count} chars</span>
                    <span>Word Count: {selected_chunk.word_count} words</span>
                    <span>Transcript Offset: char {selected_chunk.start_index:,}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📚 All Generated Chunks")
        filter_text = st.text_input(
            "Filter chunks by keyword (optional)",
            key="chunk_filter_query",
            placeholder="Type a keyword to filter chunks...",
        ).strip()

        displayed_chunks = (
            [c for c in chunks if filter_text.lower() in c.content.lower()]
            if filter_text
            else chunks
        )
        st.caption(f"Showing {len(displayed_chunks)} of {len(chunks)} chunks")

        chunk_list_container = st.container(height=420, border=False)
        with chunk_list_container:
            if not displayed_chunks:
                st.warning("No chunks match that keyword filter.")
            for chunk in displayed_chunks:
                body_html = (
                    highlight_query_terms(chunk.content, filter_text)
                    if filter_text
                    else html.escape(chunk.content)
                )
                st.markdown(
                    f"""
                    <div class="chunk-card">
                        <div class="chunk-card-header">
                            <span class="chunk-id-pill">{html.escape(chunk.chunk_id)}</span>
                            <a class="timestamp-pill" href="{html.escape(chunk.youtube_url)}" target="_blank" rel="noopener noreferrer">
                                ▶ {html.escape(chunk.timestamp_range)}
                            </a>
                        </div>
                        <div class="chunk-body">{body_html}</div>
                        <div class="chunk-footer">
                            <span>Size: {chunk.char_count} chars</span>
                            <span>Words: {chunk.word_count}</span>
                            <span>Start Offset: {chunk.start_index:,}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _render_embedding_and_vector_store_section() -> None:
    """Render Section D: Embedding & Vector Store Details."""
    emb_summary: Optional[EmbeddingSummary] = st.session_state.get("embedding_summary")
    vector_store: Optional[FAISSVectorStore] = st.session_state.get("vector_store")

    st.markdown(
        """
        <div class="section-header">
            <h2 class="section-title">🧠 D. Embedding &amp; Vector Store</h2>
            <span class="section-tag">Steps 4 &amp; 5</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if emb_summary is None or vector_store is None:
        st.caption("Sentence Transformer vector dimensions, sample vector representation, and FAISS index stats will appear here.")
        return

    vs_stats = vector_store.get_stats()

    col_emb, col_vs = st.columns(2, gap="large")

    with col_emb:
        with st.container(border=True):
            st.markdown("##### Sentence Transformer Embeddings")
            st.markdown(
                '<span class="status-badge badge-ready">● Embeddings Generated</span>',
                unsafe_allow_html=True,
            )
            e1, e2 = st.columns(2)
            e1.metric("Embedding Model", emb_summary.model_name)
            e2.metric("Vector Dimensions", f"{emb_summary.dimension}")
            st.caption(
                f"Total Chunks Embedded: **{emb_summary.total_embeddings}** | "
                f"Sample L2 Norm: **{emb_summary.sample_vector_norm}**"
            )
            st.markdown("**Sample Vector Preview (chunk_001, first 12 dims):**")
            preview_str = ", ".join(f"{v:+.5f}" for v in emb_summary.sample_vector_preview)
            st.markdown(
                f'<div class="vector-preview-box">[{html.escape(preview_str)}, ...]</div>',
                unsafe_allow_html=True,
            )

    with col_vs:
        with st.container(border=True):
            st.markdown("##### FAISS Vector Database")
            badge_cls = "badge-ready" if vs_stats.status == "Ready" else "badge-warning"
            st.markdown(
                f'<span class="status-badge {badge_cls}">● FAISS Status: {html.escape(vs_stats.status)}</span>',
                unsafe_allow_html=True,
            )
            v1, v2 = st.columns(2)
            v1.metric("Stored Vectors", f"{vs_stats.total_vectors}")
            v2.metric("Index Dimensions", f"{vs_stats.dimension}")
            st.caption(
                f"**Index Type:** {vs_stats.index_type}  \n"
                f"**Indexed Video ID:** `{vs_stats.video_id}`  \n"
                f"**Saved At:** `{vs_stats.persisted_path or 'In-Memory Only'}`  \n"
                f"**Created:** {vs_stats.created_at}"
            )


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    _load_custom_css()
    _init_session_state()

    settings = _render_sidebar()
    _render_hero_header()
    _render_video_input_section(settings)

    st.write("")
    _render_chat_and_similarity_section(settings)
    _render_transcript_section()
    _render_chunk_explorer_section()
    _render_embedding_and_vector_store_section()


if __name__ == "__main__":
    main()
