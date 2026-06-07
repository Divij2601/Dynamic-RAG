"""
Dynamic-RAG — Streamlit UI
Production-grade interface for the Dynamic-RAG system.

Run with:
    streamlit run app.py
"""

import time
import uuid
from datetime import datetime
from typing import Optional

import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

ROUTE_COLORS = {
    "internal_rag":     "#4F8EF7",
    "web_research":     "#F7C948",
    "hybrid":           "#A855F7",
    "memory":           "#22C55E",
    "direct_generation":"#F97316",
    "unknown":          "#94A3B8",
}

ROUTE_LABELS = {
    "internal_rag":     "📚 Internal RAG",
    "web_research":     "🌐 Web Research",
    "hybrid":           "🔀 Hybrid",
    "memory":           "🧠 Memory",
    "direct_generation":"⚡ Direct",
    "unknown":          "❓ Unknown",
}

STATUS_COLORS = {
    "success":   "#22C55E",
    "abstained": "#F59E0B",
    "error":     "#EF4444",
}


# ─────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dynamic-RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Global */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0F172A;
    border-right: 1px solid #1E293B;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #F1F5F9 !important; }

/* Chat bubbles */
.user-bubble {
    background: #1E40AF;
    color: #EFF6FF;
    padding: 0.75rem 1rem;
    border-radius: 1rem 1rem 0 1rem;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 0.95rem;
    line-height: 1.5;
}
.bot-bubble {
    background: #1E293B;
    border: 1px solid #334155;
    color: #E2E8F0;
    padding: 1rem 1.25rem;
    border-radius: 1rem 1rem 1rem 0;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
}
.bot-bubble.abstained {
    border-color: #D97706;
    background: #1C1007;
}
.bot-bubble.error {
    border-color: #DC2626;
    background: #1C0707;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-right: 0.35rem;
    margin-bottom: 0.4rem;
}
.badge-route { background: #1E3A5F; color: #93C5FD; border: 1px solid #1D4ED8; }
.badge-success { background: #064E3B; color: #6EE7B7; border: 1px solid #059669; }
.badge-abstained { background: #451A03; color: #FCD34D; border: 1px solid #D97706; }
.badge-error { background: #450A0A; color: #FCA5A5; border: 1px solid #DC2626; }

/* Score bars */
.score-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.3rem 0;
    font-size: 0.82rem;
    color: #94A3B8;
}
.score-label { width: 120px; flex-shrink: 0; }
.score-bar-bg {
    flex: 1;
    height: 6px;
    background: #1E293B;
    border-radius: 3px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #4F8EF7, #A855F7);
}
.score-value { width: 36px; text-align: right; color: #E2E8F0; font-weight: 500; }

/* Source card */
.source-card {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 0.5rem;
    padding: 0.6rem 0.8rem;
    margin: 0.35rem 0;
    font-size: 0.8rem;
}
.source-card .source-title {
    font-weight: 600;
    color: #93C5FD;
    margin-bottom: 0.25rem;
}
.source-card .source-meta { color: #64748B; margin-bottom: 0.3rem; }
.source-card .source-text { color: #94A3B8; line-height: 1.5; }

/* Metric card */
.metric-card {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 0.75rem;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-card .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #F1F5F9;
}
.metric-card .metric-label {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Doc card */
.doc-card {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 0.5rem;
    padding: 0.55rem 0.8rem;
    margin: 0.3rem 0;
    font-size: 0.8rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.doc-name { color: #93C5FD; font-weight: 500; max-width: 70%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.doc-chunks { color: #64748B; font-size: 0.72rem; }

/* Divider */
.rag-divider { border: none; border-top: 1px solid #1E293B; margin: 0.75rem 0; }

/* Eval table */
.eval-row {
    display: flex;
    justify-content: space-between;
    padding: 0.3rem 0;
    border-bottom: 1px solid #1E293B;
    font-size: 0.8rem;
}
.eval-row:last-child { border-bottom: none; }
.eval-metric { color: #94A3B8; }
.eval-val { color: #E2E8F0; font-weight: 500; }

/* Retry warning */
.retry-warn {
    background: #451A03;
    border: 1px solid #D97706;
    border-radius: 0.5rem;
    padding: 0.4rem 0.75rem;
    color: #FCD34D;
    font-size: 0.78rem;
    margin: 0.4rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "session_id": f"session_{uuid.uuid4().hex[:8]}",
        "messages": [],        # [{role, content, meta}]
        "api_ok": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────

def api_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        return r.json()
    except Exception:
        return {"status": "unreachable"}


def api_chat(query: str, session_id: str) -> dict:
    r = requests.post(
        f"{API_BASE}/chat/query",
        json={"query": query, "session_id": session_id},
        timeout=120
    )
    r.raise_for_status()
    return r.json()


def api_sources(query_id: str) -> dict:
    try:
        r = requests.get(
            f"{API_BASE}/query/{query_id}/sources",
            timeout=10
        )
        return r.json() if r.ok else {}
    except Exception:
        return {}


def api_session(session_id: str) -> dict:
    try:
        r = requests.get(
            f"{API_BASE}/chat/{session_id}",
            timeout=10
        )
        return r.json() if r.ok else {}
    except Exception:
        return {}


def api_metrics() -> dict:
    try:
        r = requests.get(
            f"{API_BASE}/system/metrics",
            timeout=10
        )
        return r.json() if r.ok else {}
    except Exception:
        return {}


def api_upload(file_bytes: bytes, filename: str) -> dict:
    r = requests.post(
        f"{API_BASE}/documents/upload",
        files={"file": (filename, file_bytes)},
        timeout=300
    )
    r.raise_for_status()
    return r.json()


def api_corpus_docs() -> list:
    """Pull indexed document names from corpus description."""
    try:
        from src.knowledge.corpus_description import (
            corpus_description_builder
        )
        desc = corpus_description_builder.get_description()
        docs = []
        for line in desc.split("\n"):
            if line.startswith("- ") and ": " in line:
                name = line[2:line.index(": ")]
                docs.append(name)
        return docs
    except Exception:
        return []


def get_all_sessions() -> list:
    """Fetch all sessions from MongoDB, newest first."""
    try:
        from src.memory.store import conversation_store
        return conversation_store.get_all_sessions()
    except Exception:
        return []


def load_session_messages(session_id: str) -> list:
    """Reconstruct messages list from stored turns + traces."""
    try:
        from src.memory.store import conversation_store
        return conversation_store.load_session_messages(
            session_id
        )
    except Exception:
        return []


def rename_session_db(session_id: str, name: str):
    """Persist a custom session name to MongoDB."""
    try:
        from src.memory.store import conversation_store
        conversation_store.rename_session(session_id, name)
    except Exception:
        pass


def _time_ago(dt) -> str:
    """Return a human-readable relative timestamp string."""
    if dt is None:
        return ""
    try:
        now = datetime.utcnow()
        diff = now - dt.replace(tzinfo=None)
        s = int(diff.total_seconds())
        if s < 60:
            return "just now"
        if s < 3600:
            return f"{s // 60}m ago"
        if s < 86400:
            return f"{s // 3600}h ago"
        return f"{s // 86400}d ago"
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────

def score_bar_html(label: str, value: Optional[float]) -> str:
    if value is None:
        return f"""
        <div class="score-row">
            <span class="score-label">{label}</span>
            <span style="color:#475569;font-size:0.78rem">N/A</span>
        </div>"""
    pct = int(value * 100)
    color = (
        "#22C55E" if value >= 0.8 else
        "#F59E0B" if value >= 0.5 else
        "#EF4444"
    )
    return f"""
    <div class="score-row">
        <span class="score-label">{label}</span>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct}%;background:{color}"></div>
        </div>
        <span class="score-value">{value:.2f}</span>
    </div>"""


def route_badge(route: str) -> str:
    label = ROUTE_LABELS.get(route, route)
    return f'<span class="badge badge-route">{label}</span>'


def status_badge(status: str) -> str:
    cls = {
        "success":   "badge-success",
        "abstained": "badge-abstained",
        "error":     "badge-error",
    }.get(status, "badge-route")
    icons = {
        "success": "✓ Success",
        "abstained": "⚠ Abstained",
        "error": "✗ Error",
    }
    return f'<span class="badge {cls}">{icons.get(status, status)}</span>'


def render_sources(
    sources: list,
    source_type_filter: str = None,
    global_offset: int = 0
):
    """
    Render sources numbered as 'Source N' to match
    how the LLM cited them in the answer text.
    global_offset: when showing web sources after
    document sources, pass len(doc_sources) so the
    numbers continue correctly.
    """

    if not sources:
        st.caption("No sources available.")
        return

    filtered = [
        (orig_idx, s) for orig_idx, s in enumerate(sources)
        if source_type_filter is None
        or s.get("source_type") == source_type_filter
    ]

    if not filtered:
        st.caption("No sources of this type.")
        return

    for orig_idx, s in filtered:
        n = orig_idx + 1 + global_offset
        stype = s.get("source_type", "document")
        icon = "📄" if stype == "document" else "🌐"

        if stype == "document":
            meta_obj = s.get("metadata") or {}
            filename = (
                meta_obj.get("filename")
                or s.get("source_id", "")
                or s.get("chunk_id", f"source_{n}")
            )
            filename = filename.split("/")[-1]
            page = s.get("page")
            page_str = f", page {page}" if page else ""
            expander_label = (
                f"{icon} Source {n} — {filename}{page_str}"
            )
            meta_line = (
                f"Score: {s.get('score', 0):.3f}"
                + (f"  ·  Page {page}" if page else "")
            )
        else:
            meta_obj = s.get("metadata") or {}
            web_title = meta_obj.get("title") or ""
            url = meta_obj.get("url") or s.get("source_id", "")
            display = web_title or url[:60] or f"Web source {n}"
            expander_label = f"{icon} Source {n} — {display}"
            meta_line = url[:90] if url else ""

        with st.expander(expander_label, expanded=False):
            if meta_line:
                st.caption(meta_line)
            chunk_text = s.get("text", "")
            if chunk_text:
                st.markdown(
                    f"<div style='color:#94A3B8;font-size:0.82rem;"
                    f"line-height:1.6;background:#0F172A;padding:"
                    f"0.5rem 0.75rem;border-radius:0.4rem'>"
                    f"{chunk_text[:600]}"
                    f"{'...' if len(chunk_text) > 600 else ''}</div>",
                    unsafe_allow_html=True
                )


def render_message(msg: dict):
    role = msg["role"]
    content = msg["content"]
    meta = msg.get("meta", {})

    if role == "user":
        st.markdown(
            f'<div class="user-bubble">{content}</div>',
            unsafe_allow_html=True
        )
        return

    # Bot message
    status = meta.get("status", "success")
    bubble_class = (
        "abstained" if status == "abstained" else
        "error" if status == "error" else
        ""
    )

    # Header badges
    route = meta.get("route", "unknown")
    badges = route_badge(route) + status_badge(status)

    retry = meta.get("retry_count", 0)
    if retry and int(retry) > 0:
        badges += f'<span class="badge badge-abstained">↻ {retry} retry</span>'

    st.markdown(
        f'<div style="font-size:0.72rem;margin-bottom:0.3rem">'
        f'{badges}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="bot-bubble {bubble_class}">{content}</div>',
        unsafe_allow_html=True
    )

    # Expandable detail panels
    conf = meta.get("confidence")
    faith = meta.get("faithfulness_score")
    grounded = meta.get("grounded")
    latency = meta.get("latency_ms")
    query_id = meta.get("query_id")
    sources = meta.get("sources", [])

    col_detail, col_src = st.columns([1, 1])

    with col_detail:
        with st.expander("📊 Details", expanded=False):
            st.markdown(
                score_bar_html("Confidence", conf)
                + score_bar_html("Faithfulness", faith),
                unsafe_allow_html=True
            )
            detail_cols = st.columns(2)
            with detail_cols[0]:
                st.metric(
                    "Grounded",
                    "Yes" if grounded else
                    "No" if grounded is False else "N/A"
                )
            with detail_cols[1]:
                st.metric(
                    "Latency",
                    f"{latency:.0f}ms" if latency else "N/A"
                )
            if query_id:
                st.caption(f"Query ID: `{query_id}`")

    with col_src:
        with st.expander(
            f"📎 Sources ({len(sources)})", expanded=False
        ):
            if sources:
                doc_sources = [
                    s for s in sources
                    if s.get("source_type") == "document"
                ]
                web_sources = [
                    s for s in sources
                    if s.get("source_type") == "web"
                ]
                if doc_sources and web_sources:
                    tab_d, tab_w = st.tabs(
                        [f"Documents ({len(doc_sources)})",
                         f"Web ({len(web_sources)})"]
                    )
                    with tab_d:
                        # Docs: Source 1..N
                        render_sources(
                            sources,
                            source_type_filter="document",
                            global_offset=0
                        )
                    with tab_w:
                        # Web: Source N+1..M
                        render_sources(
                            sources,
                            source_type_filter="web",
                            global_offset=len(doc_sources)
                        )
                else:
                    render_sources(sources)
            else:
                st.caption("No sources for this route.")

    st.markdown("<hr class='rag-divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────

with st.sidebar:

    # Logo + title
    st.markdown("""
    <div style="padding:0.5rem 0 1.5rem">
      <div style="font-size:1.5rem;font-weight:700;
                  color:#F1F5F9;letter-spacing:-0.02em">
        ⚡ Dynamic-RAG
      </div>
      <div style="font-size:0.72rem;color:#475569;
                  margin-top:0.15rem">
        Adaptive · Grounded · Observable
      </div>
    </div>
    """, unsafe_allow_html=True)

    # API status
    health = api_health()
    h_status = health.get("status", "unreachable")
    h_color = "#22C55E" if h_status == "healthy" else (
        "#F59E0B" if h_status == "degraded" else "#EF4444"
    )
    h_dot = "●"
    st.markdown(
        f'<div style="font-size:0.78rem;margin-bottom:1.2rem">'
        f'<span style="color:{h_color}">{h_dot}</span>'
        f' API {h_status}</div>',
        unsafe_allow_html=True
    )

    # ── Active session name / rename ──────────────────
    st.markdown("### 💬 Active Session")

    # Editable session name (stored in session_state;
    # saved to MongoDB on change).
    if "session_name" not in st.session_state:
        st.session_state.session_name = (
            st.session_state.session_id
        )

    new_name = st.text_input(
        "Session name",
        value=st.session_state.session_name,
        label_visibility="collapsed",
        placeholder="Session name…",
        key="sidebar_name_input"
    )

    if new_name != st.session_state.session_name:
        st.session_state.session_name = new_name
        rename_session_db(
            st.session_state.session_id,
            new_name
        )

    col_new, col_clear = st.columns(2)
    with col_new:
        if st.button("＋ New", use_container_width=True):
            # Keep the old session in MongoDB (it's
            # already persisted); just start fresh.
            st.session_state.session_id = (
                f"session_{uuid.uuid4().hex[:8]}"
            )
            st.session_state.session_name = (
                st.session_state.session_id
            )
            st.session_state.messages = []
            st.rerun()
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Session history list ───────────────────────────
    st.markdown("<hr class='rag-divider'>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-weight:600;color:#F1F5F9;'
        'font-size:0.9rem;margin-bottom:0.5rem">'
        '🕑 All Sessions</div>',
        unsafe_allow_html=True
    )

    all_sessions = get_all_sessions()
    current_sid = st.session_state.session_id

    if not all_sessions:
        st.caption("No sessions yet.")
    else:
        for sess in all_sessions:
            sid = sess["session_id"]
            name = sess.get("name") or sid
            preview = sess.get("preview", "")
            msg_count = sess.get("message_count", 0)
            last_active = sess.get("last_active")
            ago = _time_ago(last_active)
            is_current = (sid == current_sid)

            # Active session styling
            dot_color = "#4F8EF7" if is_current else "#334155"
            name_color = "#F1F5F9" if is_current else "#94A3B8"
            bg = (
                "background:#1E293B;border:1px solid #334155;"
                if is_current
                else "background:#0F172A;border:1px solid #1E293B;"
            )

            # Truncate name for display
            disp_name = (
                name[:28] + "…"
                if len(name) > 28
                else name
            )

            card_html = (
                f'<div style="{bg}border-radius:0.5rem;'
                f'padding:0.5rem 0.65rem;margin-bottom:0.35rem">'
                f'<div style="display:flex;align-items:center;'
                f'gap:0.4rem;margin-bottom:0.15rem">'
                f'<span style="color:{dot_color};font-size:0.55rem">●</span>'
                f'<span style="color:{name_color};font-weight:600;'
                f'font-size:0.82rem">{disp_name}</span>'
                f'</div>'
                f'<div style="color:#475569;font-size:0.72rem;'
                f'white-space:nowrap;overflow:hidden;'
                f'text-overflow:ellipsis">{preview}</div>'
                f'<div style="color:#334155;font-size:0.68rem;'
                f'margin-top:0.15rem">'
                f'{msg_count} msgs · {ago}</div>'
                f'</div>'
            )

            st.markdown(card_html, unsafe_allow_html=True)

            if not is_current:
                if st.button(
                    "Open",
                    key=f"open_sess_{sid}",
                    use_container_width=True
                ):
                    # Switch to this session —
                    # restore its messages from MongoDB.
                    st.session_state.session_id = sid
                    st.session_state.session_name = (
                        name if name != sid else sid
                    )
                    st.session_state.messages = (
                        load_session_messages(sid)
                    )
                    st.rerun()

    st.markdown("<hr class='rag-divider'>", unsafe_allow_html=True)

    # Document upload
    st.markdown("### 📁 Documents")

    uploaded = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

    if uploaded is not None:
        if st.button(
            "⬆ Ingest document",
            use_container_width=True,
            type="primary"
        ):
            with st.spinner(
                f"Ingesting {uploaded.name}…"
            ):
                try:
                    result = api_upload(
                        uploaded.getvalue(),
                        uploaded.name
                    )
                    st.success(
                        f"✓ {result.get('message', 'Indexed')}"
                    )
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    # Indexed documents
    docs = api_corpus_docs()
    if docs:
        st.caption(f"{len(docs)} documents indexed")
        for d in docs[:12]:
            name_short = (
                d[:32] + "…" if len(d) > 32 else d
            )
            st.markdown(
                f'<div class="doc-card">'
                f'<span class="doc-name" title="{d}">'
                f'📄 {name_short}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        if len(docs) > 12:
            st.caption(f"+{len(docs) - 12} more")

    st.markdown("<hr class='rag-divider'>", unsafe_allow_html=True)
    st.caption("Dynamic-RAG v1.0 · Anthropic + Groq")


# ─────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────

tab_chat, tab_metrics, tab_history, tab_eval = st.tabs([
    "💬 Chat",
    "📈 Metrics",
    "🕑 Session History",
    "🔬 Evaluation"
])


# ─────────────────────────────────────────────────────────
# TAB 1 — Chat
# ─────────────────────────────────────────────────────────

with tab_chat:

    st.markdown(
        '<h2 style="font-size:1.4rem;font-weight:600;'
        'color:#F1F5F9;margin-bottom:0.25rem">Chat</h2>'
        '<p style="color:#64748B;font-size:0.85rem;'
        'margin-bottom:1.5rem">Ask questions about your '
        'documents or anything the system can research.</p>',
        unsafe_allow_html=True
    )

    # Chat history
    if st.session_state.messages:
        for msg in st.session_state.messages:
            render_message(msg)
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 0;color:#334155">
            <div style="font-size:2.5rem;margin-bottom:0.5rem">⚡</div>
            <div style="font-size:1rem;font-weight:500;color:#94A3B8">
                Ask Dynamic-RAG anything
            </div>
            <div style="font-size:0.8rem;color:#475569;margin-top:0.4rem">
                Try: "When did the Cold War end?" or
                "Upload a document and ask about it"
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Query input
    with st.form("chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            query = st.text_input(
                "query",
                placeholder="Ask a question… (Enter to send)",
                label_visibility="collapsed"
            )
        with col_btn:
            submitted = st.form_submit_button(
                "Send ➤",
                type="primary",
                use_container_width=True
            )

    if submitted and query.strip():

        # Add user message immediately
        st.session_state.messages.append({
            "role": "user",
            "content": query.strip()
        })

        with st.spinner("Thinking…"):
            try:
                t0 = time.time()
                resp = api_chat(
                    query.strip(),
                    st.session_state.session_id
                )
                latency = (time.time() - t0) * 1000

                # Fetch sources by query_id
                sources = []
                qid = resp.get("query_id")
                if qid:
                    src_data = api_sources(qid)
                    sources = src_data.get("sources", [])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": resp.get("answer", ""),
                    "meta": {
                        "route": resp.get("route", "unknown"),
                        "status": resp.get("status", "success"),
                        "confidence": resp.get("confidence"),
                        "faithfulness_score":
                            resp.get("faithfulness_score"),
                        "grounded": resp.get("grounded"),
                        "latency_ms": latency,
                        "query_id": qid,
                        "sources": sources,
                        "retry_count": resp.get("retry_count", 0),
                    }
                })

            except Exception as exc:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Request failed: {exc}",
                    "meta": {"status": "error"}
                })

        st.rerun()


# ─────────────────────────────────────────────────────────
# TAB 2 — Metrics
# ─────────────────────────────────────────────────────────

with tab_metrics:

    st.markdown(
        '<h2 style="font-size:1.4rem;font-weight:600;'
        'color:#F1F5F9;margin-bottom:1.5rem">System Metrics</h2>',
        unsafe_allow_html=True
    )

    metrics = api_metrics()

    if not metrics or metrics.get("status") == "unavailable":
        st.info("No requests traced yet. Send a query first.")
    else:
        total = metrics.get("total_requests", 0)
        mean_lat = metrics.get("mean_latency_ms", 0)
        p95_lat = metrics.get("p95_latency_ms", 0)
        abstention = metrics.get("abstention_rate", 0)
        total_cost = metrics.get("total_cost_usd", 0)
        mean_cost = metrics.get("mean_cost_per_query_usd", 0)

        # KPI row
        k1, k2, k3, k4, k5 = st.columns(5)
        kpis = [
            (k1, str(total), "Total Queries"),
            (k2, f"{mean_lat:.0f}ms", "Mean Latency"),
            (k3, f"{p95_lat:.0f}ms", "P95 Latency"),
            (k4, f"{abstention:.1%}", "Abstention Rate"),
            (k5, f"${total_cost:.4f}", "Total Cost"),
        ]
        for col, val, label in kpis:
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{val}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts row
        c1, c2 = st.columns(2)

        route_dist = metrics.get("route_distribution", {})

        with c1:
            st.markdown(
                '<div style="font-weight:600;color:#F1F5F9;'
                'margin-bottom:0.75rem">Route Distribution</div>',
                unsafe_allow_html=True
            )
            if route_dist:
                labels = [
                    ROUTE_LABELS.get(r, r)
                    for r in route_dist
                ]
                values = list(route_dist.values())
                colors = [
                    ROUTE_COLORS.get(r, "#94A3B8")
                    for r in route_dist
                ]
                fig = go.Figure(go.Pie(
                    labels=labels,
                    values=values,
                    marker_colors=colors,
                    hole=0.55,
                    textfont_size=11
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#94A3B8",
                    legend=dict(
                        font_color="#94A3B8",
                        bgcolor="rgba(0,0,0,0)"
                    ),
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=260
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
            else:
                st.caption("No route data yet.")

        with c2:
            st.markdown(
                '<div style="font-weight:600;color:#F1F5F9;'
                'margin-bottom:0.75rem">Latency Overview</div>',
                unsafe_allow_html=True
            )
            lat_labels = ["Mean", "P95", "P99"]
            lat_vals = [
                mean_lat,
                p95_lat,
                metrics.get("p99_latency_ms", 0)
            ]
            fig2 = go.Figure(go.Bar(
                x=lat_labels,
                y=lat_vals,
                marker_color=["#4F8EF7", "#A855F7", "#EF4444"],
                text=[f"{v:.0f}ms" for v in lat_vals],
                textposition="outside",
                textfont=dict(color="#94A3B8", size=11)
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94A3B8",
                yaxis=dict(
                    title="ms",
                    gridcolor="#1E293B",
                    showgrid=True
                ),
                xaxis=dict(showgrid=False),
                margin=dict(t=30, b=10, l=10, r=10),
                height=260,
                showlegend=False
            )
            st.plotly_chart(
                fig2,
                use_container_width=True,
                config={"displayModeBar": False}
            )

        # Additional metrics
        st.markdown("<hr class='rag-divider'>", unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            st.markdown(
                f'<div style="color:#F1F5F9;font-weight:600;'
                f'margin-bottom:0.5rem">Cost Breakdown</div>'
                f'<div class="eval-row">'
                f'<span class="eval-metric">Total cost</span>'
                f'<span class="eval-val">${total_cost:.6f}</span></div>'
                f'<div class="eval-row">'
                f'<span class="eval-metric">Per query (avg)</span>'
                f'<span class="eval-val">${mean_cost:.6f}</span></div>'
                f'<div class="eval-row">'
                f'<span class="eval-metric">Total retries</span>'
                f'<span class="eval-val">'
                f'{metrics.get("total_retries", 0)}</span></div>',
                unsafe_allow_html=True
            )
        with a2:
            st.markdown(
                f'<div style="color:#F1F5F9;font-weight:600;'
                f'margin-bottom:0.5rem">Quality Signals</div>'
                f'<div class="eval-row">'
                f'<span class="eval-metric">Abstention rate</span>'
                f'<span class="eval-val">'
                f'{abstention:.1%}</span></div>'
                f'<div class="eval-row">'
                f'<span class="eval-metric">Mean confidence</span>'
                f'<span class="eval-val">'
                f'{metrics.get("mean_confidence", 0):.3f}</span></div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────
# TAB 3 — Session History
# ─────────────────────────────────────────────────────────

with tab_history:

    st.markdown(
        '<h2 style="font-size:1.4rem;font-weight:600;'
        'color:#F1F5F9;margin-bottom:1rem">Session History</h2>',
        unsafe_allow_html=True
    )

    hist_sid = st.text_input(
        "Session ID to load",
        value=st.session_state.session_id,
        key="hist_sid"
    )

    if st.button("Load session", type="primary"):
        with st.spinner("Fetching history…"):
            hist = api_session(hist_sid)

        msgs = hist.get("messages", [])
        count = hist.get("message_count", 0)

        st.markdown(
            f'<div style="color:#64748B;font-size:0.82rem;'
            f'margin-bottom:1rem">'
            f'{count} turns in session `{hist_sid}`</div>',
            unsafe_allow_html=True
        )

        if msgs:
            for m in msgs:
                q = m.get("query", "")
                a = m.get("answer", "")
                route = m.get("route", "unknown")
                conf = m.get("confidence", None)
                ts = m.get("timestamp", "")

                st.markdown(
                    f'<div class="user-bubble">{q}</div>',
                    unsafe_allow_html=True
                )
                badges = route_badge(route)
                if ts:
                    badges += (
                        f'<span style="color:#475569;'
                        f'font-size:0.72rem;margin-left:0.5rem">'
                        f'{ts[:19]}</span>'
                    )
                st.markdown(
                    f'<div style="font-size:0.72rem;'
                    f'margin-bottom:0.25rem">{badges}</div>',
                    unsafe_allow_html=True
                )
                if conf is not None:
                    badges += (
                        f' <span style="color:#64748B;'
                        f'font-size:0.72rem">conf {conf:.2f}</span>'
                    )
                st.markdown(
                    f'<div class="bot-bubble">{a}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<hr class='rag-divider'>",
                    unsafe_allow_html=True
                )
        else:
            st.info(
                "No history found for this session. "
                "Start a conversation in the Chat tab."
            )


# ─────────────────────────────────────────────────────────
# TAB 4 — Evaluation
# ─────────────────────────────────────────────────────────

with tab_eval:

    st.markdown(
        '<h2 style="font-size:1.4rem;font-weight:600;'
        'color:#F1F5F9;margin-bottom:0.5rem">'
        'Evaluation Dashboard</h2>'
        '<p style="color:#64748B;font-size:0.85rem;'
        'margin-bottom:1.5rem">Run the benchmark suite '
        'or view the latest saved report.</p>',
        unsafe_allow_html=True
    )

    # Load latest report
    import glob
    import json
    import os

    report_files = sorted(
        glob.glob("evaluation/reports/*.json"),
        reverse=True
    )

    sel_report = None

    if report_files:
        report_names = [
            os.path.basename(f) for f in report_files
        ]
        chosen = st.selectbox(
            "Report",
            report_names,
            index=0
        )
        chosen_path = os.path.join(
            "evaluation/reports", chosen
        )
        with open(chosen_path, encoding="utf-8") as f:
            sel_report = json.load(f)

    if sel_report:

        ts = sel_report.get("timestamp", "")[:19]
        ds = sel_report.get("dataset", "")

        st.markdown(
            f'<div style="color:#64748B;font-size:0.78rem;'
            f'margin-bottom:1.5rem">'
            f'📅 {ts} &nbsp;·&nbsp; 📂 {os.path.basename(ds)}'
            f'</div>',
            unsafe_allow_html=True
        )

        p1 = sel_report.get("plane_1_retrieval", {})
        gc = sel_report.get("gate_c_routing", {})
        p2 = sel_report.get("plane_2_generation", {})
        p3 = sel_report.get("plane_3_system", {})

        # ── Plane 1
        with st.expander("📡 Plane 1 — Retrieval Quality", expanded=True):
            cols = st.columns(3)
            r_metrics = [
                ("Recall@K",          p1.get("Recall@K")),
                ("MRR",               p1.get("MRR")),
                ("NDCG@K",            p1.get("NDCG@K")),
                ("Hit Rate",          p1.get("Hit Rate")),
                ("Context Precision", p1.get("Context Precision")),
                ("Context Recall",    p1.get("Context Recall")),
            ]
            for col, (name, val) in zip(
                cols * 2, r_metrics
            ):
                with col:
                    delta = (
                        "excellent" if val and val >= 0.9 else
                        "good" if val and val >= 0.7 else
                        "needs work" if val else None
                    )
                    st.metric(
                        name,
                        f"{val:.4f}" if val is not None else "N/A",
                        delta=delta,
                        delta_color="normal"
                    )

            # Radar chart
            if all(v for _, v in r_metrics):
                cats = [n for n, _ in r_metrics] + [r_metrics[0][0]]
                vals = [v for _, v in r_metrics] + [r_metrics[0][1]]
                fig_r = go.Figure(go.Scatterpolar(
                    r=vals, theta=cats,
                    fill="toself",
                    line_color="#4F8EF7",
                    fillcolor="rgba(79,142,247,0.15)"
                ))
                fig_r.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            gridcolor="#1E293B",
                            color="#475569"
                        ),
                        bgcolor="rgba(0,0,0,0)"
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#94A3B8",
                    height=300,
                    margin=dict(t=20, b=20)
                )
                st.plotly_chart(
                    fig_r,
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

        # ── Gate C
        with st.expander("🗺 Gate C — Routing Accuracy"):
            ra = gc.get("Route Accuracy")
            st.markdown(
                score_bar_html("Route Accuracy", ra),
                unsafe_allow_html=True
            )
            per_class = gc.get("Per-Class Accuracy", {})
            for route, acc in per_class.items():
                label = ROUTE_LABELS.get(route, route)
                st.markdown(
                    score_bar_html(label, acc),
                    unsafe_allow_html=True
                )

            confusion = gc.get("Confusion Matrix", {})
            if confusion:
                st.markdown(
                    '<div style="color:#64748B;font-size:0.78rem;'
                    'margin-top:0.75rem;margin-bottom:0.35rem">'
                    'Confusion Matrix</div>',
                    unsafe_allow_html=True
                )
                rows = sorted(confusion.keys())
                all_preds = sorted(
                    {p for d in confusion.values()
                     for p in d}
                )
                z = [
                    [confusion.get(r, {}).get(c, 0)
                     for c in all_preds]
                    for r in rows
                ]
                y_labels = [
                    ROUTE_LABELS.get(r, r) for r in rows
                ]
                x_labels = [
                    ROUTE_LABELS.get(c, c) for c in all_preds
                ]
                fig_cm = go.Figure(go.Heatmap(
                    z=z, x=x_labels, y=y_labels,
                    colorscale="Blues",
                    showscale=False,
                    text=z,
                    texttemplate="%{text}",
                    textfont_size=13
                ))
                fig_cm.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#94A3B8",
                    height=220,
                    margin=dict(t=10, b=10, l=10, r=10),
                    xaxis_title="Predicted",
                    yaxis_title="Expected"
                )
                st.plotly_chart(
                    fig_cm,
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

        # ── Plane 2
        with st.expander("✍️ Plane 2 — Generation Quality"):
            g_metrics = [
                ("Faithfulness",    p2.get("Faithfulness")),
                ("Groundedness",    p2.get("Groundedness")),
                ("Answer Relevance",p2.get("Answer Relevance")),
                ("Completeness",    p2.get("Completeness")),
                ("Citation Accuracy",p2.get("Citation Accuracy")),
            ]
            bars = "".join(
                score_bar_html(n, v) for n, v in g_metrics
            )
            st.markdown(bars, unsafe_allow_html=True)
            st.caption(
                f"Evaluated on "
                f"{p2.get('Evaluated (answerable)', '?')} "
                f"answerable queries."
            )

        # ── Plane 3
        with st.expander("⚙️ Plane 3 — System Quality"):
            s_cols = st.columns(3)
            s_metrics = [
                ("E2E Accuracy",  p3.get("End-to-End Accuracy")),
                ("Rejection Rate",p3.get("Rejection Rate")),
                ("Mean Latency",  p3.get("Mean Latency (ms)")),
                ("P95 Latency",   p3.get("P95 Latency (ms)")),
                ("Failure Count", p3.get("Failure Count")),
                ("Cost/Query",    p3.get("Estimated Cost / Query")),
            ]
            for col, (name, val) in zip(
                s_cols * 2, s_metrics
            ):
                with col:
                    if val is None:
                        display = "N/A"
                    elif "ms" in name.lower():
                        display = f"{val:.0f}ms"
                    elif name == "Failure Count":
                        display = str(int(val))
                    elif isinstance(val, float) and val <= 1:
                        display = f"{val:.4f}"
                    else:
                        display = str(val)
                    st.metric(name, display)

    else:
        st.info(
            "No benchmark reports found. "
            "Run the benchmark to generate one:\n\n"
            "```bash\n"
            "python -m evaluation.runner "
            "evaluation/data/test_set.json\n```"
        )
