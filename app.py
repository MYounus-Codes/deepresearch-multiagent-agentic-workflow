import html
import re
import time

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #ffffff !important;
    color: #0d0d0d !important;
}

.stApp { background: #ffffff !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 0 !important; max-width: 100% !important; margin: 0 !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
section[data-testid="stMain"] { overflow: hidden !important; }

/* ── Navbar ── */
.rm-navbar {
    height: 56px; border-bottom: 1px solid #e5e5e5;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 28px; background: #fff !important;
    position: sticky; top: 0; z-index: 200;
}
.rm-logo { font-size: 16px; font-weight: 600; color: #111 !important; letter-spacing: -0.4px; }
.rm-logo span { color: #f97316 !important; }
.rm-badge { font-size: 11px; color: #6b7280 !important; background: #f3f4f6 !important; border-radius: 999px; padding: 3px 10px; }

/* ── Force text color everywhere in app ── */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em { color: #1a1a1a !important; }

/* ── st.container(height=...) scrollable box ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    background: transparent !important;
}

/* scrollbar styling for st.container(height) */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    scrollbar-width: thin;
    scrollbar-color: #d1d5db transparent;
}
[data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar { width: 5px; }
[data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-track { background: transparent; }
[data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }

/* columns layout */
[data-testid="stHorizontalBlock"] { gap: 0 !important; align-items: stretch !important; }
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { padding: 0 !important; }

/* sidebar column background */
.rm-sidebar-col { background: #fafafa; height: 100%; }

/* ── EMPTY STATE — centered hero input ── */
.rm-hero {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    min-height: calc(100vh - 160px);
    padding: 0 24px;
    text-align: center;
    background: #fff;
}
.rm-hero-icon {
    width: 52px; height: 52px; border-radius: 16px;
    background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
    border: 1px solid #fed7aa;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; margin-bottom: 24px;
}
.rm-hero-title {
    font-size: 28px; font-weight: 600; color: #0d0d0d !important;
    letter-spacing: -0.5px; margin-bottom: 10px; line-height: 1.25;
}
.rm-hero-title span { color: #f97316 !important; }
.rm-hero-sub {
    font-size: 15px; color: #6b7280 !important;
    max-width: 420px; line-height: 1.65; margin-bottom: 36px;
}
.rm-hero-agents {
    display: flex; gap: 8px; margin-bottom: 36px; flex-wrap: wrap; justify-content: center;
}
.rm-agent-tag {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; font-weight: 500; color: #374151 !important;
    background: #f9fafb !important; border: 1px solid #e5e7eb;
    border-radius: 999px; padding: 5px 12px;
}
.rm-agent-dot { width: 6px; height: 6px; border-radius: 50%; background: #f97316; }
.rm-agent-dot.blue { background: #6366f1; }
.rm-agent-dot.green { background: #10b981; }
.rm-agent-dot.amber { background: #f59e0b; }

/* hero input box */
.rm-hero-input-box {
    width: 100%; max-width: 600px;
    background: #fff; border: 1.5px solid #e5e7eb;
    border-radius: 16px; padding: 4px 6px 4px 18px;
    display: flex; align-items: center; gap: 8px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    transition: border-color 0.15s, box-shadow 0.15s;
}
.rm-hero-input-box:focus-within {
    border-color: #f97316;
    box-shadow: 0 2px 20px rgba(249,115,22,0.12);
}

/* hero example chips */
.rm-hero-chips { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-top: 20px; }
.rm-hero-chip-label { font-size: 11px; color: #9ca3af !important; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }

/* ── CHAT messages area ── */
.rm-msg-pad { padding: 28px 0 12px; }

.rm-user-row {
    display: flex; justify-content: flex-end;
    padding: 0 28px; margin-bottom: 16px;
}
.rm-user-bubble {
    background: #f3f4f6 !important; color: #111 !important;
    border-radius: 18px 18px 4px 18px;
    padding: 11px 16px; font-size: 14px; line-height: 1.6;
    max-width: 70%; font-weight: 400;
}

.rm-ai-row {
    display: flex; gap: 11px; align-items: flex-start;
    padding: 0 28px; margin-bottom: 16px;
}
.rm-ai-avatar {
    width: 28px; height: 28px; flex-shrink: 0;
    background: #f97316 !important; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 600; color: #fff !important; margin-top: 1px;
}
.rm-ai-avatar.critic-av { background: #6366f1 !important; }
.rm-ai-content { flex: 1; min-width: 0; }
.rm-ai-label { font-size: 10px; font-weight: 600; color: #9ca3af !important; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 6px; }

.rm-ai-body { font-size: 14px; line-height: 1.78; color: #1a1a1a !important; }
.rm-ai-body h1 { font-size: 19px; font-weight: 600; margin: 16px 0 6px; color: #0d0d0d !important; }
.rm-ai-body h2 { font-size: 15px; font-weight: 600; margin: 14px 0 5px; color: #0d0d0d !important; }
.rm-ai-body h3 { font-size: 14px; font-weight: 600; margin: 12px 0 4px; color: #0d0d0d !important; }
.rm-ai-body p { margin: 0 0 10px; color: #1a1a1a !important; }
.rm-ai-body ul, .rm-ai-body ol { margin: 6px 0 10px 18px; padding: 0; }
.rm-ai-body li { margin-bottom: 5px; color: #374151 !important; }
.rm-ai-body strong { font-weight: 600; color: #0d0d0d !important; }
.rm-ai-body a { color: #f97316 !important; }
.rm-ai-body code { font-size: 12px; background: #f3f4f6 !important; padding: 2px 5px; border-radius: 4px; color: #374151 !important; }
.rm-ai-body hr { border: none; border-top: 1px solid #e5e5e5; margin: 14px 0; }

/* override Streamlit markdown inside ai body */
.rm-ai-body [data-testid="stMarkdownContainer"] p,
.rm-ai-body [data-testid="stMarkdownContainer"] li,
.rm-ai-body [data-testid="stMarkdownContainer"] strong { color: #1a1a1a !important; }

/* spinner */
.rm-status-row { display: flex; gap: 10px; align-items: center; padding: 2px 28px 14px; }
.rm-spinner {
    width: 14px; height: 14px; flex-shrink: 0;
    border: 2px solid #f3f4f6; border-top-color: #f97316;
    border-radius: 50%; animation: rm-spin 0.75s linear infinite;
}
@keyframes rm-spin { to { transform: rotate(360deg); } }
.rm-status-text { font-size: 13px; color: #9ca3af !important; font-style: italic; }

/* pills */
.rm-pipeline-pills { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 28px 14px; }
.rm-pill { display: inline-flex; align-items: center; gap: 5px; font-size: 11px; font-weight: 500; padding: 4px 10px; border-radius: 999px; border: 1px solid; }
.rm-pill-done { background: #f0fdf4 !important; border-color: #bbf7d0 !important; color: #166534 !important; }
.rm-pill-run  { background: #fff7ed !important; border-color: #fed7aa !important; color: #9a3412 !important; }

/* ── Bottom input bar (shown when chat active) ── */
.rm-input-bar {
    border-top: 1px solid #e5e5e5;
    padding: 12px 28px 16px;
    background: #fff !important;
}
.rm-chips-label { font-size: 10px; color: #9ca3af !important; margin-bottom: 8px; font-weight: 500; letter-spacing: 0.4px; text-transform: uppercase; }

/* ── Sidebar ── */
.rm-sb-header {
    padding: 14px 18px 11px; border-bottom: 1px solid #e5e5e5;
    background: #fafafa !important;
}
.rm-sb-title { font-size: 11px; font-weight: 600; color: #9ca3af !important; letter-spacing: 0.6px; text-transform: uppercase; }
.rm-sb-body { padding: 14px 14px; background: #fafafa !important; }

.rm-step-label { display: flex; align-items: center; gap: 6px; font-size: 10px; font-weight: 600; color: #9ca3af !important; letter-spacing: 0.4px; text-transform: uppercase; margin: 6px 0 7px; }
.rm-step-dot { width: 6px; height: 6px; border-radius: 50%; background: #f97316; flex-shrink: 0; }
.rm-step-dot.green { background: #10b981; }

.rm-source { background: #fff !important; border: 1px solid #e5e5e5; border-radius: 9px; padding: 9px 11px; margin-bottom: 7px; }
.rm-source-title { font-size: 11px; font-weight: 600; color: #111 !important; margin-bottom: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rm-source-url { font-size: 10px; color: #f97316 !important; display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-decoration: none; margin-bottom: 3px; }
.rm-source-snippet { font-size: 10px; color: #6b7280 !important; line-height: 1.5; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }

.rm-scraped { background: #f0fdf4 !important; border: 1px solid #bbf7d0; border-radius: 7px; padding: 7px 11px; font-size: 10px; color: #166534 !important; margin-bottom: 7px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rm-note { font-size: 10px; color: #9ca3af !important; background: #f9fafb !important; border: 1px solid #f3f4f6; border-radius: 7px; padding: 6px 11px; margin-bottom: 7px; }
.rm-divider { height: 1px; background: #f0f0f0; margin: 8px 0; }

/* ── Input widget overrides ── */
.stTextInput > div > div > input {
    background: #f9fafb !important; border: 1.5px solid #e5e5e5 !important;
    border-radius: 12px !important; color: #111 !important;
    font-family: 'Inter', sans-serif !important; font-size: 14px !important;
    padding: 11px 14px !important; caret-color: #f97316 !important; box-shadow: none !important;
}
.stTextInput > div > div > input:focus {
    border-color: #f97316 !important; background: #fff !important;
    box-shadow: 0 0 0 3px rgba(249,115,22,0.1) !important;
}
.stTextInput > div > div > input::placeholder { color: #9ca3af !important; }
.stTextInput > label { display: none !important; }
div[data-testid="stTextInput"] { margin-bottom: 0 !important; }

.st-key-send_btn > div > button, .st-key-send_btn_hero > div > button {
    background: #f97316 !important; color: #fff !important; border: none !important;
    border-radius: 10px !important; font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important; font-size: 13px !important;
    padding: 10px 18px !important; height: 44px !important;
    box-shadow: none !important; white-space: nowrap !important;
}
.st-key-send_btn > div > button:hover,
.st-key-send_btn_hero > div > button:hover { background: #ea6c0a !important; }

.st-key-clear_btn > div > button {
    background: transparent !important; color: #9ca3af !important;
    border: 1.5px solid #e5e5e5 !important; border-radius: 10px !important;
    font-size: 13px !important; padding: 10px 12px !important;
    height: 44px !important; box-shadow: none !important;
}

.st-key-chip_0 > div > button,
.st-key-chip_1 > div > button,
.st-key-chip_2 > div > button,
.st-key-chip_h0 > div > button,
.st-key-chip_h1 > div > button,
.st-key-chip_h2 > div > button {
    background: #fff !important; border: 1.5px solid #e5e5e5 !important;
    color: #6b7280 !important; border-radius: 999px !important;
    font-family: 'Inter', sans-serif !important; font-size: 11px !important;
    font-weight: 400 !important; padding: 5px 13px !important; box-shadow: none !important;
}
.st-key-chip_0 > div > button:hover,
.st-key-chip_1 > div > button:hover,
.st-key-chip_2 > div > button:hover,
.st-key-chip_h0 > div > button:hover,
.st-key-chip_h1 > div > button:hover,
.st-key-chip_h2 > div > button:hover {
    border-color: #f97316 !important; color: #f97316 !important; background: #fff7ed !important;
}

/* Download */
.stDownloadButton > div > button {
    background: transparent !important; border: 1.5px solid #e5e5e5 !important;
    color: #6b7280 !important; border-radius: 8px !important;
    font-size: 12px !important; box-shadow: none !important; padding: 5px 12px !important;
}
.stDownloadButton > div > button:hover { border-color: #f97316 !important; color: #f97316 !important; }

div[data-testid="stStatusWidget"] { display: none !important; }
div[data-testid="stVerticalBlock"] > div { padding: 0 !important; }
div.element-container { margin: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _tool_outputs(messages, tool_name=None):
    out = []
    for m in messages:
        if isinstance(m, ToolMessage) and (tool_name is None or getattr(m, "name", None) == tool_name):
            out.append(m.content)
    return out

def _tool_call_args(messages, tool_name=None):
    out = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                if tool_name is None or tc.get("name") == tool_name:
                    out.append(tc.get("args", {}))
    return out

def parse_search_results(raw: str):
    items = []
    for block in raw.split("\n---\n"):
        block = block.strip()
        if not block:
            continue
        title   = re.search(r"Title:\s*(.*)", block)
        url     = re.search(r"URL:\s*(.*)", block)
        snippet = re.search(r"Snippet\s*:\s*(.*)", block, re.DOTALL)
        if url:
            items.append({
                "title":   (title.group(1).strip() if title else "Untitled")[:180],
                "url":     url.group(1).strip(),
                "snippet": (snippet.group(1).strip() if snippet else "")[:380],
            })
    seen, deduped = set(), []
    for it in items:
        if it["url"] not in seen:
            seen.add(it["url"])
            deduped.append(it)
    return deduped


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("chat", []), ("activity", []), ("_pipeline_step", 0), ("_pipeline_data", {})]:
    if k not in st.session_state:
        st.session_state[k] = v

def _clear():
    for k in ("chat", "activity"):
        st.session_state[k] = []
    st.session_state["_q"] = ""
    st.session_state["_pipeline_step"] = 0
    st.session_state["_pipeline_data"] = {}

def _chip(ex):
    st.session_state["_q"] = ex
    st.session_state["_auto"] = True


EXAMPLES = ["LLM agents 2025", "CRISPR gene editing", "Fusion energy"]
has_chat  = bool(st.session_state.chat)
has_activity = bool(st.session_state.activity)

# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rm-navbar">
    <div class="rm-logo">Research<span>Mind</span></div>
    <div class="rm-badge">Multi-agent AI · 4 specialized agents</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# EMPTY STATE — full-width centered hero
# ══════════════════════════════════════════════════════════════
if not has_chat:
    st.markdown("""
    <div class="rm-hero">
        <div class="rm-hero-icon">🔬</div>
        <div class="rm-hero-title">What do you want to <span>research?</span></div>
        <div class="rm-hero-sub">
            Four specialized AI agents will search the web, scrape sources,
            write a structured report, and critique it — all in seconds.
        </div>
        <div class="rm-hero-agents">
            <span class="rm-agent-tag"><span class="rm-agent-dot"></span>Search Agent</span>
            <span class="rm-agent-tag"><span class="rm-agent-dot blue"></span>Reader Agent</span>
            <span class="rm-agent-tag"><span class="rm-agent-dot green"></span>Writer Agent</span>
            <span class="rm-agent-tag"><span class="rm-agent-dot amber"></span>Critic Agent</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # centered input — constrain with columns
    _, center_col, _ = st.columns([1, 3, 1])
    with center_col:
        hi_col, hb_col = st.columns([8, 2], gap="small")
        with hi_col:
            query = st.text_input("q", placeholder="Ask anything to research…",
                                  key="_q", label_visibility="collapsed")
        with hb_col:
            send = st.button("Research →", key="send_btn_hero", use_container_width=True)

        st.markdown('<div class="rm-hero-chip-label">Try an example</div>', unsafe_allow_html=True)
        hc0, hc1, hc2, _hsp = st.columns([1, 1, 1, 0.1])
        for col, ex, i in zip((hc0, hc1, hc2), EXAMPLES, range(3)):
            col.button(ex, key=f"chip_h{i}", on_click=_chip, args=(ex,), use_container_width=False)


# ══════════════════════════════════════════════════════════════
# CHAT + SIDEBAR layout
# ══════════════════════════════════════════════════════════════
else:
    if has_activity:
        chat_col, sidebar_col = st.columns([62, 38], gap="small")
    else:
        chat_col = st.container()
        sidebar_col = None

    # ── CHAT ──────────────────────────────────────────────────
    with chat_col:
        # scrollable messages — st.container with fixed height
        VIEWPORT_H = 580   # px — adjust if your screen is taller/shorter
        msg_box = st.container(height=VIEWPORT_H, border=False)
        with msg_box:
            st.markdown('<div class="rm-msg-pad">', unsafe_allow_html=True)
            for msg in st.session_state.chat:
                role = msg["role"]

                if role == "user":
                    st.markdown(f"""
                    <div class="rm-user-row">
                        <div class="rm-user-bubble">{html.escape(msg["content"])}</div>
                    </div>""", unsafe_allow_html=True)

                elif role == "spinner":
                    st.markdown(f"""
                    <div class="rm-status-row">
                        <div class="rm-spinner"></div>
                        <span class="rm-status-text">{msg["content"]}</span>
                    </div>""", unsafe_allow_html=True)

                elif role == "pills":
                    st.markdown(f'<div class="rm-pipeline-pills">{msg["content"]}</div>',
                                unsafe_allow_html=True)

                elif role == "assistant":
                    kind = msg.get("type", "report")
                    if kind == "report":
                        av_cls, label, initials = "", "Research report", "RM"
                    else:
                        av_cls, label, initials = "critic-av", "Critic review", "CR"

                    st.markdown(f"""
                    <div class="rm-ai-row">
                        <div class="rm-ai-avatar {av_cls}">{initials}</div>
                        <div class="rm-ai-content">
                            <div class="rm-ai-label">{label}</div>
                            <div class="rm-ai-body">""", unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="font-size:14px;line-height:1.78;color:#1a1a1a">{msg["content"]}</div>',
                        unsafe_allow_html=True)
                    st.markdown("</div></div></div>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # download
        report_md = next((m["content"] for m in st.session_state.chat if m.get("type") == "report"), None)
        if report_md:
            dl_col, _ = st.columns([1, 4])
            with dl_col:
                st.download_button("⬇ Download report", data=report_md,
                                   file_name=f"report_{int(time.time())}.md", mime="text/markdown")

        # sticky input bar
        st.markdown('<div class="rm-input-bar">', unsafe_allow_html=True)
        st.markdown('<div class="rm-chips-label">Try an example</div>', unsafe_allow_html=True)
        c0, c1, c2, _sp = st.columns([1, 1, 1, 2])
        for col, ex, i in zip((c0, c1, c2), EXAMPLES, range(3)):
            col.button(ex, key=f"chip_{i}", on_click=_chip, args=(ex,), use_container_width=False)

        ic, bc, cc = st.columns([10, 1.6, 0.8], gap="small")
        with ic:
            query = st.text_input("q", placeholder="Ask anything to research…",
                                  key="_q", label_visibility="collapsed")
        with bc:
            send = st.button("Research →", key="send_btn", use_container_width=True)
        with cc:
            st.button("↺", key="clear_btn", on_click=_clear, use_container_width=True, help="Clear")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────
    if has_activity and sidebar_col is not None:
        with sidebar_col:
            st.markdown("""
            <div class="rm-sb-header">
                <div class="rm-sb-title">Activity Feed</div>
            </div>""", unsafe_allow_html=True)

            # scrollable sidebar body via st.container(height)
            sb_box = st.container(height=VIEWPORT_H + 50, border=False)
            with sb_box:
                st.markdown('<div class="rm-sb-body">', unsafe_allow_html=True)
                for item in st.session_state.activity:
                    kind = item["kind"]
                    if kind == "search_results":
                        n = len(item["data"])
                        st.markdown(f'<div class="rm-step-label"><div class="rm-step-dot"></div> Web sources found ({n})</div>', unsafe_allow_html=True)
                        for src in item["data"]:
                            t = html.escape(src["title"])
                            u = html.escape(src["url"])
                            s = html.escape(src["snippet"])
                            st.markdown(f"""
                            <div class="rm-source">
                                <div class="rm-source-title">{t}</div>
                                <a class="rm-source-url" href="{u}" target="_blank">{u}</a>
                                <div class="rm-source-snippet">{s}</div>
                            </div>""", unsafe_allow_html=True)
                    elif kind == "scraped":
                        u = html.escape(item["data"] or "—")
                        st.markdown(f'<div class="rm-step-label"><div class="rm-step-dot green"></div> Page scraped</div><div class="rm-scraped">🔗 {u}</div>', unsafe_allow_html=True)
                    elif kind == "note":
                        st.markdown(f'<div class="rm-note">{html.escape(item["data"])}</div>', unsafe_allow_html=True)
                    elif kind == "divider":
                        st.markdown('<div class="rm-divider"></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════
# Collect send trigger from whichever button was rendered
send_hero = st.session_state.get("send_btn_hero", False)
trigger = (locals().get("send") or send_hero) or st.session_state.pop("_auto", False)
current_query = st.session_state.get("_q", "").strip()

if trigger and not current_query:
    st.warning("Please enter a research topic.")
    trigger = False

if trigger and current_query:
    st.session_state.chat = []
    st.session_state.activity = []
    st.session_state["_pipeline_step"] = 1
    st.session_state["_pipeline_data"] = {"query": current_query}
    st.session_state.chat.append({"role": "user", "content": current_query})
    st.session_state.chat.append({"role": "spinner", "content": "Search Agent — scanning the web…"})
    st.rerun()

step = st.session_state.get("_pipeline_step", 0)
data = st.session_state.get("_pipeline_data", {})

if step == 1:
    q = data["query"]
    try:
        agent = build_search_agent()
        result = agent.invoke({"messages": [("user", f"Find recent, reliable and detailed information about: {q}")]})
    except Exception as e:
        st.session_state.chat = [c for c in st.session_state.chat if c["role"] != "spinner"]
        st.session_state.chat.append({"role": "pills", "content": f'<span class="rm-pill rm-pill-run">✗ Search failed: {html.escape(str(e))}</span>'})
        st.session_state["_pipeline_step"] = 0
        st.rerun()
        st.stop()
    raw_outputs = _tool_outputs(result["messages"], "search_web")
    items = []
    for raw in raw_outputs:
        items.extend(parse_search_results(raw))
    final_text = result["messages"][-1].content
    raw_combined = "\n---\n".join(raw_outputs) if raw_outputs else final_text
    data.update({"raw_combined": raw_combined, "search_items": items})
    st.session_state["_pipeline_data"] = data
    st.session_state.activity += [{"kind": "search_results", "data": items}, {"kind": "divider", "data": None}]
    st.session_state.chat = [c for c in st.session_state.chat if c["role"] != "spinner"]
    st.session_state.chat.append({"role": "spinner", "content": "Reader Agent — scraping top source…"})
    st.session_state["_pipeline_step"] = 2
    st.rerun()

elif step == 2:
    q, raw_combined = data["query"], data["raw_combined"]
    try:
        agent = build_reader_agent()
        result = agent.invoke({"messages": [("user",
            f"Based on the following search results about '{q}', pick the most relevant URL and scrape it.\n\nSearch Results:\n{raw_combined[:800]}")]})
    except Exception as e:
        st.session_state.chat = [c for c in st.session_state.chat if c["role"] != "spinner"]
        st.session_state.chat.append({"role": "pills", "content": f'<span class="rm-pill rm-pill-run">✗ Reader failed: {html.escape(str(e))}</span>'})
        st.session_state["_pipeline_step"] = 0
        st.rerun()
        st.stop()
    scrape_calls = _tool_call_args(result["messages"], "scrape_url")
    scraped_url  = scrape_calls[0].get("url") if scrape_calls else None
    tool_outs    = _tool_outputs(result["messages"], "scrape_url")
    scraped_text = tool_outs[0] if tool_outs else result["messages"][-1].content
    data.update({"scraped_text": scraped_text, "scraped_url": scraped_url})
    st.session_state["_pipeline_data"] = data
    st.session_state.activity += [{"kind": "scraped", "data": scraped_url}, {"kind": "divider", "data": None}]
    st.session_state.chat = [c for c in st.session_state.chat if c["role"] != "spinner"]
    st.session_state.chat.append({"role": "spinner", "content": "Writer — drafting the report…"})
    st.session_state["_pipeline_step"] = 3
    st.rerun()

elif step == 3:
    q, raw_combined, scraped_text = data["query"], data["raw_combined"], data["scraped_text"]
    try:
        report = writer_chain.invoke({"topic": q, "research": f"SEARCH RESULTS:\n{raw_combined}\n\nSCRAPED CONTENT:\n{scraped_text}"})
    except Exception as e:
        st.session_state.chat = [c for c in st.session_state.chat if c["role"] != "spinner"]
        st.session_state.chat.append({"role": "pills", "content": f'<span class="rm-pill rm-pill-run">✗ Writer failed: {html.escape(str(e))}</span>'})
        st.session_state["_pipeline_step"] = 0
        st.rerun()
        st.stop()
    data["report"] = report
    st.session_state["_pipeline_data"] = data
    st.session_state.activity += [{"kind": "note", "data": "Report drafted — see chat ↗"}, {"kind": "divider", "data": None}]
    st.session_state.chat = [c for c in st.session_state.chat if c["role"] != "spinner"]
    st.session_state.chat.append({"role": "spinner", "content": "Critic — reviewing the report…"})
    st.session_state["_pipeline_step"] = 4
    st.rerun()

elif step == 4:
    report = data["report"]
    try:
        feedback = critic_chain.invoke({"report": report})
    except Exception as e:
        feedback = f"Critic unavailable: {e}"
    n = len(data.get("search_items", []))
    st.session_state.activity.append({"kind": "note", "data": "Critic review done — see chat ↗"})
    st.session_state.chat = [c for c in st.session_state.chat if c["role"] != "spinner"]
    st.session_state.chat.append({
        "role": "pills",
        "content": (
            f'<span class="rm-pill rm-pill-done">✓ Search — {n} source{"s" if n != 1 else ""}</span>'
            f'<span class="rm-pill rm-pill-done">✓ Reader — content extracted</span>'
            f'<span class="rm-pill rm-pill-done">✓ Writer — report drafted</span>'
            f'<span class="rm-pill rm-pill-done">✓ Critic — review complete</span>'
        )
    })
    st.session_state.chat.append({"role": "assistant", "type": "report",  "content": report})
    st.session_state.chat.append({"role": "assistant", "type": "critic",  "content": feedback})
    st.session_state["_pipeline_step"] = 0
    st.session_state["_pipeline_data"] = {}
    st.rerun()