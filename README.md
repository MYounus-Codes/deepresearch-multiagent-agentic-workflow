# 🔬 ResearchMind — Multi-Agent AI Research Assistant

ResearchMind is an end-to-end, multi-agent research pipeline that autonomously searches the web, scrapes relevant content, writes a structured report, and critiques its own output — all powered by a single LLM backend (Groq). It provides both a **Streamlit web UI** and a **CLI pipeline**.

---

## ✨ Features

- **4 Specialized Agents** — Search, Reader, Writer, and Critic work sequentially to produce polished research reports.
- **Autonomous Web Research** — Uses the Tavily search API to find relevant, recent sources.
- **Deep Content Extraction** — Scrapes full text from the most relevant URL using BeautifulSoup.
- **Structured Reports** — Generates professional reports with Introduction, Key Findings, Conclusion, and Sources.
- **Self-Critique** — A dedicated Critic agent scores the report (X/10) and provides actionable improvement suggestions.
- **Streamlit UI** (`app.py`) — Modern chat interface with real-time pipeline pills, activity feed sidebar, and report download.
- **CLI Pipeline** (`pipeline.py`) — Lightweight terminal-based execution for scripting or quick use.
- **Fully Local Orchestration** — No external orchestration services; the entire pipeline runs via LangChain + LangChain Classic agents.

---

## 🧠 Key Points

| Aspect | Detail |
|---|---|
| **LLM** | `qwen/qwen3-32b` via Groq (temperature 0) |
| **Search** | Tavily API — 5 results per query |
| **Scraping** | `requests` + BeautifulSoup — 3000 chars max |
| **Agent Framework** | LangChain Classic (`create_agent`) for tool-using agents; LCEL chains for Writer & Critic |
| **UI** | Streamlit with custom CSS, Inter font, orange accent theme |
| **CLI** | Sequential execution with rich console output |
| **State Management** | Pipeline step-driven via `st.session_state` (UI) or dict (CLI) |
| **Python** | 3.12+ |

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM Provider** | [Groq](https://groq.com/) (`qwen/qwen3-32b`) |
| **Agent Framework** | LangChain, LangChain Classic, LangChain Community |
| **Web Search** | [Tavily](https://tavily.com/) (`tavily`, `langchain-tavily`) |
| **Web Scraping** | `requests` + `BeautifulSoup4` |
| **UI** | [Streamlit](https://streamlit.io/) |
| **Configuration** | `python-dotenv` + `.env` |
| **Package Manager** | `uv` |
| **Output** | Markdown reports (downloadable from UI) |

---

## 📁 Folder Structure

```
deepresearch-multiagent-project/
├── agents.py              # Agent definitions (Search, Reader, Writer, Critic)
├── app.py                 # Streamlit web UI (ResearchMind)
├── pipeline.py            # CLI research pipeline
├── tools.py               # LangChain tools (search_web, scrape_url)
├── main.py                # Placeholder entry point
├── output.md              # Sample pipeline output (2026 FIFA World Cup)
├── pyproject.toml         # Package spec + dependencies
├── uv.lock                # Resolved dependency lockfile
├── .env                   # API keys (TAVILY + GROQ)
├── .gitignore             # Python ignores + .env
├── .python-version        # 3.12
└── README.md              # This file
```

---

## 🔄 Agent Flow

```
USER QUERY
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SEARCH AGENT (tool: search_web via Tavily)                        │
│  "Find recent, reliable info about {topic}"                        │
│  Output: 5 results with Title, URL, Snippet                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  READER AGENT (tool: scrape_url via requests + BeautifulSoup)      │
│  "Pick the most relevant URL and scrape it for deeper content"     │
│  Output: Clean text from one URL (up to 3000 chars)                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  WRITER CHAIN (pure LLM, no tools)                                 │
│  Synthesizes search results + scraped content into a structured    │
│  report: Introduction, Key Findings (3+), Conclusion, Sources      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CRITIC CHAIN (pure LLM, no tools)                                 │
│  Reviews the report and produces:                                  │
│  - Score: X/10                                                     │
│  - Strengths                                                       │
│  - Areas to Improve                                                │
│  - One-line verdict                                                │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                   FINAL OUTPUT
              (Report + Critic Feedback)
```

### Execution Details

1. **Search Agent** — A LangChain Classic agent with access to the `search_web` tool. It invokes Tavily to fetch 5 search results (title, URL, snippet) for the topic.
2. **Reader Agent** — A LangChain Classic agent with access to the `scrape_url` tool. It receives the first 800 characters of the search results, selects the most relevant URL, and extracts clean text via HTTP GET + BeautifulSoup (stripping scripts, styles, nav, footer; max 3000 chars).
3. **Writer Chain** — An LCEL chain (`prompt | llm | StrOutputParser`). It takes the full search results + scraped content and produces a professional markdown report.
4. **Critic Chain** — An LCEL chain that evaluates the report, assigns a score out of 10, lists strengths and weaknesses, and gives a one-line verdict.

All four agents share the same `ChatGroq(model="qwen/qwen3-32b", temperature=0)` instance.

### Dual Entry Points

| Entry Point | Command | Use Case |
|---|---|---|
| **Streamlit UI** | `streamlit run app.py` | Interactive web interface with activity sidebar, download support |
| **CLI Pipeline** | `python pipeline.py` | Terminal-based execution for quick research or automation |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys for [Groq](https://console.groq.com/keys) and [Tavily](https://app.tavily.com/)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd deepresearch-multiagent-project

# Create .env and add your API keys
echo "TAVILY_API_KEY=your_tavily_key" > .env
echo "GROQ_API_KEY=your_groq_key" >> .env

# Install dependencies (with uv)
uv sync

# Or with pip
pip install -e .
```

### Usage

```bash
# Web UI
streamlit run app.py

# CLI
python pipeline.py
```

---

## 📋 Sample Output

See [`output.md`](./output.md) for a complete run on the topic *"2026 FIFA World Cup economic impact"*, including search results, scraped content, the generated report, and critic feedback (Score: 7/10).

---

## ⚠️ Security Note

The `.env` file containing API keys was committed to git history in early commits. While `.gitignore` now excludes `.env`, the keys remain in the repository's history. Rotate any exposed keys immediately.

---

## 📄 License

MIT
