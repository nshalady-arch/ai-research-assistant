# AI Research Assistant

An agentic AI system that finds the most essential **recent** research in a field (default:
**Agentic AI**), evaluates it against an expert rubric, and writes you a briefing:

1. a 1-page **Executive Brief** with a **TL;DR**, on how to architect a multi-agent system
   using this year's standards,
2. a **reading list** of up to 10 papers, each with authors, source, date, a direct link, *why it
   made the list* and the *key takeaway*,
3. an appendix showing how every candidate scored.

The result is a Markdown file: `reports/ai-research-<YYYY-MM-DD>.md`.

The system follows the structure of the [CAPSTONE_CM sample project](https://github.com/jaby2323/CAPSTONE_CM)
(LLM client, vector store, tools and agents as separate modules) and is built from a research-librarian
prompt that is split between two agents (see [How the prompt is used](#how-the-research-prompt-is-used)).

---

## How it works

```
                        research_agent_orchestrator.py
                                     │
             1. make sure the vector DB exists (create it if needed)
                                     │
   ┌─────────────────────────────────▼─────────────────────────────────┐
   │ A1  COLLECTOR  (collector_agent.py), a Claude tool-use agent      │
   │                                                                   │
   │   search_arxiv ───────── arXiv API (cs.AI/CL/MA/LG/SE)            │
   │   get_trending_papers ── Hugging Face Daily Papers (+ upvotes)    │
   │   get_lab_blog_posts ─── OpenAI, DeepMind, Google Research, BAIR, │
   │                          HF, Amazon Science, LangChain, MIT News  │
   │   check_discussion_buzz  Hacker News (points / comments)          │
   │   store_candidates ───── fetch full text → chunk → embed → FAISS  │
   └─────────────────────────────────┬─────────────────────────────────┘
                 2. wait for A1 to finish (12–30 candidates stored)
                                     │
                       ┌─────────────▼─────────────┐
                       │  vector_db/ (FAISS index, │
                       │  chunks, paper metadata)  │
                       └─────────────┬─────────────┘
   ┌─────────────────────────────────▼─────────────────────────────────┐
   │ A2  PROCESSOR  (processor_agent.py)                               │
   │   • drop anything published outside the window                    │
   │   • per paper: retrieve its key chunks (RAG) → Claude scores it   │
   │     1-10 on 4 criteria + type + "why" + "key takeaway" (parallel) │
   │   • weighted ranking → quality bar (no padding) → breadth check   │
   │     (≥1 survey, ≥1 empirical, ≥1 reframing when available)        │
   │   • retrieve excerpts from the selected papers → Claude writes    │
   │     the TL;DR + Executive Brief                                   │
   │   • render Markdown → tools.write_markdown_report                 │
   └─────────────────────────────────┬─────────────────────────────────┘
                                     ▼
                     reports/ai-research-<YYYY-MM-DD>.md
```

**Guardrails in code (not only in the prompt):**

- A1 can only store items that a search tool actually returned. Every title, author list, date and
  URL in the report comes from the source API, never from model output, so links can't be invented.
- The 14-day window is applied twice: by the tools during collection and again by A2 before scoring.
- "Strictly limit to 10": the list is capped at `TOP_N`. "Don't pad": items below
  `MIN_QUALITY_SCORE` are dropped, and the report says so when fewer than 10 qualify.
- Breadth of coverage is enforced when ranking. If no qualifying item of a required type exists,
  the report says so.
- If A1 stops early or stores too few items, the pool is topped up from the items it already saw,
  ranked by objective signals (HF upvotes, HN points, venue acceptance).
- If one paper's evaluation fails, it is listed as excluded and the rest of the run continues.

### Files

| File | Purpose |
|---|---|
| `research_agent_orchestrator.py` | **Entry point.** Ensures the DB exists, runs A1, waits, runs A2, prints where the report is. |
| `collector_agent.py` | **A1 Collector**: Claude tool-use loop over the internet tools; stores the candidate pool in FAISS. |
| `processor_agent.py` | **A2 Processor**: RAG-based evaluation, ranking, Executive Brief, report rendering. |
| `llm_client.py` | Provider-agnostic LLM interface: `complete()`, `complete_structured()` (validated Pydantic output), `run_tool_loop()`. Two backends: `ClaudeClient` (Anthropic SDK, with adaptive thinking, effort, refusal fallback and prompt caching) and `OpenAICompatibleClient` (open models via Ollama, LM Studio, Groq, OpenRouter and similar). `create_llm_client()` picks one from config. |
| `vector_store.py` | FAISS vector store used by A1 and A2: create/open/clear/delete, `add_paper()`, `list_papers()`, `search()` (optionally limited to certain papers). Includes the embedders. |
| `tools.py` | Tools: arXiv, HF Daily Papers, lab-blog RSS, Hacker News, full-text extraction, `write_markdown_report()`, plus the Claude tool schemas for A1. |
| `prompts.py` | The research prompt split into reusable fragments, and the system/user prompts for each agent. |
| `config.py` | All settings: domain, window, model, effort, sources, weights, paths. |
| `scripts/create_vector_db.py` | Create the vector DB if it doesn't exist. |
| `scripts/clear_vector_db.py` | Empty the DB between runs (keeps the DB). |
| `scripts/delete_vector_db.py` | Delete the DB directory entirely. |

---

## Setup

**Requirements:** Python 3.10+ (tested on 3.14), internet access, and **either** an Anthropic API
key **or** a free open-source model (see [Using a free open-source model](#using-a-free-open-source-model)).

```bash
cd AAICapstone
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**API key:** create one at <https://console.anthropic.com> (Settings → API Keys), then either:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # current shell only
# or keep it in a local file (git-ignored):
cp .env.example .env                      # then edit .env and paste your key
```

No other keys are needed: arXiv, Hugging Face, the lab-blog RSS feeds and Hacker News are all public.

**Optional: better embeddings.** By default the vector DB uses a built-in hashing embedder
(numpy only, offline, no model download). For neural embeddings run
`pip install sentence-transformers` (this pulls in PyTorch, about 1–2 GB), then delete and
recreate the DB. With the default `EMBEDDER=auto` setting it is picked up automatically.

---

## Using a free open-source model

You can swap Claude for any model served through an **OpenAI-compatible API**. Set
`LLM_PROVIDER=openai_compatible` in `.env`, or pass `--provider openai_compatible`. The agents
stay the same; only `llm_client.py` changes backend.

**Option A: run locally with [Ollama](https://ollama.com).** Free, private, nothing leaves your machine.

```bash
ollama pull llama3.1:8b                        # any model with tool-calling support
OLLAMA_CONTEXT_LENGTH=16384 ollama serve       # in a separate terminal; the larger context matters
python research_agent_orchestrator.py --provider openai_compatible --model llama3.1:8b --lite --fresh
```

- **Context length:** Ollama's default context is small. Without `OLLAMA_CONTEXT_LENGTH`, long
  prompts are silently cut off.
- **`--lite`:** shrinks tool results, the candidate pool (10 papers) and the retrieved context, and
  runs one evaluation at a time. Recommended for local models.
- **Speed:** depends on your hardware. On a CPU-only Intel i9 MacBook, `llama3.1:8b` took about
  20 s for a short text reply and about 5 min for each structured or tool-calling call. A lite run
  makes roughly 20–25 such calls, so expect **about 2 hours or more**. A GPU or Apple Silicon is
  much faster. `llama3.2:3b` is faster but less reliable.
- **Other local servers:** LM Studio works the same way with
  `OPENAI_COMPAT_BASE_URL=http://localhost:1234/v1`.

**Option B: a hosted open-model API with a free tier.** Fast, but needs a free account and key,
and free tiers are rate-limited (the client waits and retries on HTTP 429).

```bash
# .env, Groq example (OpenRouter: https://openrouter.ai/api/v1 with a ":free" model)
LLM_PROVIDER=openai_compatible
OPENAI_COMPAT_BASE_URL=https://api.groq.com/openai/v1
OPENAI_COMPAT_API_KEY=gsk_...
OPENAI_COMPAT_MODEL=llama-3.3-70b-versatile
```

**What to expect from open models:**
- Smaller models follow the multi-step collector instructions less reliably than Claude. The code
  guardrails still apply: invented IDs are rejected, and the pool is topped up from real search
  results if the agent stops early.
- Structured output is requested as a JSON schema and always validated. If a reply doesn't match,
  the model gets one retry with the validation error; papers that still fail are listed as excluded.
- `*_EFFORT` settings, prompt caching and the refusal fallback are Claude-only and are ignored here.
- The Executive Brief will be noticeably less sharp than Claude's.

---

## Running it

```bash
python research_agent_orchestrator.py
```

That's all you need. The orchestrator creates the vector DB if it's missing, runs A1, then A2,
and ends with:

```
Report written to: .../reports/ai-research-2026-10-18.md
Selected 10 of 27 evaluated papers in 6.4 min · LLM usage: 41 calls, ...
```

Useful flags:

| Command | What it does |
|---|---|
| `python research_agent_orchestrator.py --fresh` | Clear the vector DB first, so only this run's candidates are considered |
| `python research_agent_orchestrator.py --skip-collect` | Skip A1 and re-run A2 on what's already stored (e.g. after changing weights) |
| `python research_agent_orchestrator.py --days 7` | Use a 7-day window instead of 14 |
| `python research_agent_orchestrator.py --top-n 5` | Pick at most 5 papers |
| `python research_agent_orchestrator.py --domain "AI safety"` | Research a different field (the prompts adapt) |
| `python research_agent_orchestrator.py --provider openai_compatible --model llama3.1:8b --lite` | Use a free local model (see above) |
| `python research_agent_orchestrator.py -v` | Debug logging |

### Managing the vector DB

```bash
python scripts/create_vector_db.py            # create if missing (no-op if it exists)
python scripts/create_vector_db.py --embedder hashing   # force the offline embedder
python scripts/clear_vector_db.py             # empty it between runs (asks to confirm; -y to skip)
python scripts/delete_vector_db.py            # remove it completely (asks to confirm; -y to skip)
```

Without clearing, papers from earlier runs stay in the DB. A2 still ignores anything older than
the window, so this is safe, but use `--fresh` or `clear_vector_db.py` if you only want the
current run's candidates.

---

## Where to find the results

| Location | Contents |
|---|---|
| `reports/ai-research-<YYYY-MM-DD>.md` | **The report.** A second run on the same day overwrites it. |
| `vector_db/` | FAISS index (`index.faiss`), text chunks (`chunks.json`), paper metadata and signals (`papers.json`), `manifest.json` |
| Console log | A1's tool calls, every A2 score, and total token usage |

Report layout:

1. **Executive Brief**: TL;DR bullets, then about a page on multi-agent architecture, citing papers as `[#n]`
2. **Coverage Notes**: e.g. "Only 7 high-quality items were found…" or which breadth swap was made
3. **Reading List at a Glance**: table of rank, linked title, type, score and key takeaway
4. **The Top N: Details**: title, authors, source, date, URL, scores, *Why this made the top 10*, *Key takeaway*
5. **Appendix**: method, every candidate's score, and why each one wasn't selected

---

## How the research prompt is used

`prompts.py` stores the original prompt as named fragments and gives each agent only the parts it needs:

| Prompt fragment | A1 Collector | A2 Evaluator | A2 Brief writer |
|---|:-:|:-:|:-:|
| Role: *research librarian and domain expert in "Agentic AI"* | ✓ | ✓ | ✓ |
| Goal: *identify the 10 most essential articles… last 14 days* | ✓ | ✓ | ✓ |
| Criteria: source authority, recency & relevance, citation momentum, foundational impact | ✓ (screening) | ✓ (1-10 rubric) | |
| Breadth of coverage (survey / empirical / reframing) | ✓ (diverse pool) | ✓ (classify type) | ✓ |
| Per-selection fields (why top 10, key takeaway) | | ✓ | |
| *Synthesize into a 1-page Executive Brief…* | | | ✓ |
| Constraints (max 10, ≤14 days, say so instead of padding) | ✓ (window) | | ✓ |

The deterministic parts of the prompt (the 10-item cap, the 14-day limit, no padding, the breadth
mix, the metadata fields) are also enforced in code, as described above.

> The original prompt says "published in the last week" in one place and "within the last 14 days"
> in its constraints. The default is **14 days**; use `--days 7` for a strict one-week window.

---

## Configuration

All settings are in `config.py`. Most can also be set as environment variables (see `.env.example`):

| Setting | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `claude` | `claude` or `openai_compatible` (also `--provider`) |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Use `claude-sonnet-5-5` to lower the cost |
| `OPENAI_COMPAT_BASE_URL` / `OPENAI_COMPAT_MODEL` / `OPENAI_COMPAT_API_KEY` | Ollama at `localhost:11434/v1` / `llama3.1:8b` / empty | For `openai_compatible`; the model can also be set with `--model` |
| `OPENAI_COMPAT_MAX_TOKENS` / `OPENAI_COMPAT_TIMEOUT_SECONDS` | 4096 / 900 | Output cap and per-request timeout for open models |
| `RESEARCH_DOMAIN` / `LOOKBACK_DAYS` / `TOP_N` | Agentic AI / 14 / 10 | Also available as CLI flags |
| `BRIEF_FOCUS` | how to architect a multi-agent system | Topic of the Executive Brief |
| `COLLECTOR_EFFORT` / `EVALUATOR_EFFORT` / `SYNTHESIS_EFFORT` | medium / medium / high | Claude reasoning effort for each step |
| `CRITERIA_WEIGHTS` | impact 35%, authority 25%, relevance 20%, momentum 20% | Edit in `config.py` |
| `MIN_QUALITY_SCORE` | 6.0 | Quality bar on the 1-10 weighted score |
| `MAX_CANDIDATES_TO_STORE` / `MIN_CANDIDATES_TO_STORE` | 30 / 12 | Size of the A1 candidate pool |
| `EVAL_WORKERS` | 4 | Parallel A2 evaluations |
| `EMBEDDER` | auto | `auto`, `hashing`, or `st:<sentence-transformers model>` |
| `USE_REFUSAL_FALLBACK` | 1 | If the model declines a request, the API retries it on a fallback model |
| `LAB_BLOG_FEEDS`, `ARXIV_CATEGORIES` | see file | Add or remove sources |

**Cost and time (rough estimate):** one run makes about 15–25 Claude calls for A1, one call per
candidate for A2 (about 30), and one for the brief. Expect roughly 5–10 minutes and a few dollars on
Claude Opus 5.5. Prompt caching lowers A1's cost, and Sonnet 5.5 roughly halves the total. The
console prints the exact token usage at the end of each run.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No Claude credentials found` | Set `ANTHROPIC_API_KEY`, or use an open model with `--provider openai_compatible` |
| `cannot reach http://localhost:11434/v1` | Start Ollama: `OLLAMA_CONTEXT_LENGTH=16384 ollama serve` |
| `model '...' not found on the server` | Use the exact tag from `ollama list` (e.g. `llama3.1:8b`) or `ollama pull` it |
| Open-model evaluations fail schema validation | Use a larger model, or one with good JSON / tool-calling support (e.g. Qwen 3, Llama 3.1+) |
| `invalid or missing API key` | Check the key in the Anthropic Console |
| `rate limited even after retries` | Wait and re-run; lower `EVAL_WORKERS` |
| `Embedder dimension does not match` | You changed the embedder. Run `python scripts/delete_vector_db.py -y` and re-run |
| A source fails (e.g. a blog feed returns 403) | It's logged and skipped; the other sources still run |
| Fewer than 10 papers in the report | Expected when there aren't enough strong items in the window (see Coverage Notes). Lower `MIN_QUALITY_SCORE` or widen `--days` if you want more |
