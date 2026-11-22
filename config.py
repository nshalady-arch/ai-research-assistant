"""
Central configuration for the AI Research Personal Assistant.

Everything tunable lives here: the research domain, the look-back window,
model settings, data sources, vector-store paths and the A2 scoring weights.
Most values can be overridden with environment variables (see .env.example)
or with command-line flags on research_agent_orchestrator.py.
"""

import os
from pathlib import Path

try:  # Optional: load a local .env file if python-dotenv is installed.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a convenience, not a requirement
    pass

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Research task ------------------------------------------------------------
DOMAIN = os.environ.get("RESEARCH_DOMAIN", "Agentic AI")
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "14"))  # "published within the last 14 days"
TOP_N = int(os.environ.get("TOP_N", "10"))                  # "strictly limit to 10 items"
# What the 1-page Executive Brief should teach, built from the selected papers.
BRIEF_FOCUS = os.environ.get("BRIEF_FOCUS", "how to architect a multi-agent system")

# --- LLM configuration (used by llm_client.py) --------------------------------
# "claude"            -> Anthropic API (needs ANTHROPIC_API_KEY, paid per token)
# "openai_compatible" -> any OpenAI-compatible server: Ollama / LM Studio (free,
#                        local) or hosted open-model APIs (Groq, OpenRouter, ...)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "claude")

# Open-model settings (used when LLM_PROVIDER=openai_compatible).
# Default: a local Ollama server. The model must support tool calling (for A1).
OPENAI_COMPAT_BASE_URL = os.environ.get("OPENAI_COMPAT_BASE_URL", "http://localhost:11434/v1")
OPENAI_COMPAT_MODEL = os.environ.get("OPENAI_COMPAT_MODEL", "llama3.1:8b")
OPENAI_COMPAT_API_KEY = os.environ.get("OPENAI_COMPAT_API_KEY", "")  # not needed for local servers
OPENAI_COMPAT_MAX_TOKENS = int(os.environ.get("OPENAI_COMPAT_MAX_TOKENS", "4096"))
OPENAI_COMPAT_TIMEOUT_SECONDS = int(os.environ.get("OPENAI_COMPAT_TIMEOUT_SECONDS", "900"))  # CPUs are slow

# Claude settings (used when LLM_PROVIDER=claude).
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
# Server-side refusal fallback: if the model's safety classifiers decline a
# request, the API re-runs it on Anthropic's recommended fallback model.
# Set USE_REFUSAL_FALLBACK=0 to turn it off.
USE_REFUSAL_FALLBACK = os.environ.get("USE_REFUSAL_FALLBACK", "1") == "1"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Reasoning effort per agent step ("low" | "medium" | "high" | "xhigh" | "max").
COLLECTOR_EFFORT = os.environ.get("COLLECTOR_EFFORT", "medium")
EVALUATOR_EFFORT = os.environ.get("EVALUATOR_EFFORT", "medium")
SYNTHESIS_EFFORT = os.environ.get("SYNTHESIS_EFFORT", "high")
LLM_MAX_RETRIES = 3
LLM_TIMEOUT_SECONDS = 600

# --- A1 Collector: data sources ----------------------------------------------
# arXiv categories that carry most agentic-AI research.
ARXIV_CATEGORIES = ["cs.AI", "cs.CL", "cs.MA", "cs.LG", "cs.SE"]
ARXIV_MAX_RESULTS_PER_QUERY = 40
# Hugging Face Daily Papers = community curation + upvotes (citation momentum).
HF_DAILY_PAPERS_PER_DAY = 50
# Research-lab blogs (source authority for non-arXiv announcements).
LAB_BLOG_FEEDS = {
    "OpenAI": "https://openai.com/news/rss.xml",
    "Google DeepMind": "https://deepmind.google/blog/rss.xml",
    "Google Research": "https://research.google/blog/rss/",
    "Berkeley AI Research (BAIR)": "https://bair.berkeley.edu/blog/feed.xml",
    "Hugging Face": "https://huggingface.co/blog/feed.xml",
    "Amazon Science": "https://www.amazon.science/index.rss",
    "LangChain": "https://blog.langchain.com/rss/",
    "MIT News (AI)": "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",
}
# Hard caps so one run stays bounded in cost and time.
COLLECTOR_MAX_TURNS = 25          # max Claude <-> tool round trips for A1
MAX_CANDIDATES_TO_STORE = 30      # the candidate pool A1 hands to A2
MIN_CANDIDATES_TO_STORE = 12      # A1 tops up with heuristics if the agent stores fewer
MAX_FULLTEXT_CHARS = 40_000       # per-document text kept in the vector DB
TOOL_RESULT_MAX_ITEMS = 40        # items per search-tool result shown to the model
TOOL_RESULT_ABSTRACT_CHARS = 350  # abstract preview length in tool results
HTTP_TIMEOUT_SECONDS = 20
HTTP_USER_AGENT = "ai-research-assistant/1.0 (+capstone project; polite crawler)"

# --- Vector store (FAISS) -----------------------------------------------------
VECTOR_DB_DIR = Path(os.environ.get("VECTOR_DB_DIR", PROJECT_ROOT / "vector_db"))
# "auto" uses sentence-transformers if it is installed, else the built-in
# hashing embedder (no model download, works offline).
EMBEDDER = os.environ.get("EMBEDDER", "auto")
SENTENCE_TRANSFORMER_MODEL = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
HASHING_EMBEDDER_DIM = 1024
CHUNK_SIZE_CHARS = 1500
CHUNK_OVERLAP_CHARS = 200

# --- A2 Processor: evaluation & ranking ---------------------------------------
# Weights for the selection criteria from the research prompt (sum to 1.0).
CRITERIA_WEIGHTS = {
    "foundational_impact": 0.35,
    "source_authority": 0.25,
    "recency_relevance": 0.20,
    "citation_momentum": 0.20,
}
# Papers whose weighted score (1-10 scale) falls below this are not "high
# quality" and are never used to pad the top-N list.
MIN_QUALITY_SCORE = float(os.environ.get("MIN_QUALITY_SCORE", "6.0"))
# Breadth of coverage: the final list must include each of these if any
# qualifying candidate of that type exists.
REQUIRED_CONTRIBUTION_TYPES = ["survey", "empirical", "reframing"]
EVAL_CONTEXT_CHUNKS = 4           # retrieved chunks per paper shown to the evaluator
SYNTHESIS_CONTEXT_CHUNKS = 12     # retrieved chunks shown to the brief writer
EVAL_WORKERS = int(os.environ.get("EVAL_WORKERS", "4"))  # parallel evaluations

# --- Lite mode (--lite) ------------------------------------------------------
# Smaller prompts and fewer calls, for small/slow local models. Applied by the
# orchestrator on top of the values above.
LITE_OVERRIDES = {
    "MAX_CANDIDATES_TO_STORE": 10,
    "MIN_CANDIDATES_TO_STORE": 6,
    "TOOL_RESULT_MAX_ITEMS": 10,
    "TOOL_RESULT_ABSTRACT_CHARS": 200,
    "COLLECTOR_MAX_TURNS": 12,
    "EVAL_CONTEXT_CHUNKS": 2,
    "SYNTHESIS_CONTEXT_CHUNKS": 6,
    "EVAL_WORKERS": 1,
}

# --- Output -------------------------------------------------------------------
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "reports"))
