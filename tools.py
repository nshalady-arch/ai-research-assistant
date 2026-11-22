"""
Tools used by the agents: internet retrieval, text extraction and report output.

Internet sources (all public, no API keys required):
  * arXiv API              - peer-reviewed / preprint research   (source authority, recency)
  * Hugging Face Daily     - community-curated papers + upvotes  (citation momentum)
    Papers API
  * Research-lab blogs     - RSS feeds of major labs             (source authority)
  * Hacker News (Algolia)  - discussion volume for a title/topic (citation momentum)

Every function here is deterministic, plain Python; none of them call an LLM.
COLLECTOR_TOOL_SCHEMAS describes the subset A1 (Collector) can call through
Claude tool use; the Collector binds those schemas to these functions.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import feedparser
import requests

import config

log = logging.getLogger(__name__)

_session = requests.Session()
_session.headers.update({"User-Agent": config.HTTP_USER_AGENT})

# arXiv asks API clients to wait ~3 seconds between requests.
_ARXIV_MIN_INTERVAL = 3.0
_arxiv_lock = threading.Lock()
_arxiv_last_call = 0.0


# =============================================================================
# Small helpers
# =============================================================================
def window_start(days: int = config.LOOKBACK_DAYS) -> date:
    """First calendar day (UTC) inside the look-back window."""
    return (datetime.now(timezone.utc) - timedelta(days=days)).date()


def in_window(published: str | None, days: int = config.LOOKBACK_DAYS) -> bool:
    """True if an ISO date string (YYYY-MM-DD...) falls inside the window."""
    if not published:
        return False
    try:
        return date.fromisoformat(published[:10]) >= window_start(days)
    except ValueError:
        return False


def _get(url: str, params: dict[str, Any] | None = None, retries: int = 2) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = _session.get(url, params=params, timeout=config.HTTP_TIMEOUT_SECONDS)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last_exc}")


def _iso(struct_time: Any) -> str | None:
    if not struct_time:
        return None
    return datetime(*struct_time[:6], tzinfo=timezone.utc).date().isoformat()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def _arxiv_id(raw: str) -> str:
    """'http://arxiv.org/abs/2610.22341v2' -> '2610.22341'."""
    tail = raw.rstrip("/").split("/abs/")[-1]
    return re.sub(r"v\d+$", "", tail)


# =============================================================================
# Internet retrieval
# =============================================================================
def search_arxiv(query: str, days: int = config.LOOKBACK_DAYS,
                 max_results: int = config.ARXIV_MAX_RESULTS_PER_QUERY,
                 categories: list[str] | None = None) -> list[dict[str, Any]]:
    """Search arXiv titles+abstracts for `query`, newest first, limited to the window."""
    global _arxiv_last_call
    categories = categories or config.ARXIV_CATEGORIES
    phrase = query.replace('"', "").strip()
    term = f'all:"{phrase}"' if " " in phrase else f"all:{phrase}"
    cats = " OR ".join(f"cat:{c}" for c in categories)
    params = {
        "search_query": f"({term}) AND ({cats})",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": max(1, min(int(max_results), 100)),
    }
    with _arxiv_lock:  # serialise + throttle arXiv calls
        wait = _ARXIV_MIN_INTERVAL - (time.monotonic() - _arxiv_last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = _get("https://export.arxiv.org/api/query", params=params)
        finally:
            _arxiv_last_call = time.monotonic()

    feed = feedparser.parse(resp.text)
    papers = []
    for e in feed.entries:
        published = _iso(e.get("published_parsed"))
        if not in_window(published, days):
            continue
        pid = _arxiv_id(e.get("id", ""))
        cats_ = [t.get("term") for t in e.get("tags", []) if t.get("term")]
        papers.append({
            "id": pid,
            "title": _clean(e.get("title", "")),
            "authors": [a.get("name", "") for a in e.get("authors", [])],
            "published": published,
            "source": f"arXiv ({cats_[0] if cats_ else 'preprint'})",
            "source_type": "preprint",
            "url": f"https://arxiv.org/abs/{pid}",
            "pdf_url": f"https://arxiv.org/pdf/{pid}",
            "abstract": _clean(e.get("summary", "")),
            "venue_comment": _clean(e.get("arxiv_comment", "")),
            "categories": cats_,
        })
    return papers


def fetch_hf_daily_papers(days: int = config.LOOKBACK_DAYS,
                          per_day: int = config.HF_DAILY_PAPERS_PER_DAY) -> list[dict[str, Any]]:
    """Papers featured on Hugging Face Daily Papers in the window, with upvotes."""
    by_id: dict[str, dict[str, Any]] = {}
    start = window_start(days)
    today = datetime.now(timezone.utc).date()
    day = start
    while day <= today:
        try:
            items = _get("https://huggingface.co/api/daily_papers",
                         params={"date": day.isoformat(), "limit": per_day}).json()
        except Exception as exc:  # noqa: BLE001 - one bad day shouldn't stop collection
            log.warning("HF daily papers %s failed: %s", day, exc)
            items = []
        for item in items or []:
            p = item.get("paper", {})
            pid = p.get("id")
            published = (p.get("publishedAt") or item.get("publishedAt") or "")[:10]
            if not pid or not in_window(published, days):
                continue
            prev = by_id.get(pid)
            if prev and prev["hf_upvotes"] >= (p.get("upvotes") or 0):
                continue
            by_id[pid] = {
                "id": pid,
                "title": _clean(p.get("title") or item.get("title", "")),
                "authors": [a.get("name", "") for a in p.get("authors", []) if not a.get("hidden")],
                "published": published,
                "source": "arXiv (featured on Hugging Face Daily Papers)",
                "source_type": "preprint",
                "url": f"https://arxiv.org/abs/{pid}",
                "pdf_url": f"https://arxiv.org/pdf/{pid}",
                "abstract": _clean(p.get("summary") or item.get("summary", "")),
                "hf_upvotes": p.get("upvotes") or 0,
                "hf_comments": item.get("numComments") or 0,
                "github_repo": p.get("githubRepo"),
                "github_stars": p.get("githubStars") or 0,
                "keywords": p.get("ai_keywords") or [],
            }
        day += timedelta(days=1)
    return sorted(by_id.values(), key=lambda x: x["hf_upvotes"], reverse=True)


def fetch_lab_blog_posts(days: int = config.LOOKBACK_DAYS,
                         feeds: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Recent posts from research-lab blogs (RSS/Atom) inside the window."""
    posts = []
    for lab, url in (feeds or config.LAB_BLOG_FEEDS).items():
        try:
            parsed = feedparser.parse(_get(url).content)
        except Exception as exc:  # noqa: BLE001
            log.warning("feed %s failed: %s", lab, exc)
            continue
        for e in parsed.entries:
            published = _iso(e.get("published_parsed") or e.get("updated_parsed"))
            link = e.get("link")
            if not link or not in_window(published, days):
                continue
            posts.append({
                "id": "blog:" + hashlib.sha1(link.encode()).hexdigest()[:12],
                "title": _clean(e.get("title", "")),
                "authors": [e.get("author")] if e.get("author") else [lab],
                "published": published,
                "source": f"{lab} blog",
                "source_type": "blog",
                "url": link,
                "abstract": _clean(re.sub(r"<[^>]+>", " ", e.get("summary", "")))[:2000],
            })
    return posts


def check_hn_discussion(queries: list[str], days: int = config.LOOKBACK_DAYS) -> dict[str, Any]:
    """Hacker News stories about any of `queries` in the window: count, points, comments.

    Algolia matches loosely (story text, comments), so a hit only counts if a
    query string appears in the story's title or URL.
    """
    since = int(datetime.combine(window_start(days), datetime.min.time(),
                                 tzinfo=timezone.utc).timestamp())
    by_id: dict[str, dict[str, Any]] = {}
    for query in queries:
        data = _get("https://hn.algolia.com/api/v1/search", params={
            "query": query, "tags": "story",
            "numericFilters": f"created_at_i>{since}", "hitsPerPage": 10,
        }).json()
        for h in data.get("hits", []):
            haystack = f"{h.get('title') or ''} {h.get('url') or ''}".lower()
            if query.lower() in haystack:
                by_id[h.get("objectID")] = h
    hits = sorted(by_id.values(), key=lambda h: h.get("points") or 0, reverse=True)
    return {
        "queries": queries,
        "stories": len(hits),
        "total_points": sum(h.get("points") or 0 for h in hits),
        "total_comments": sum(h.get("num_comments") or 0 for h in hits),
        "top": [{"title": h.get("title"), "points": h.get("points"),
                 "url": f"https://news.ycombinator.com/item?id={h.get('objectID')}"}
                for h in hits[:3]],
    }


class _TextExtractor(HTMLParser):
    """Collects visible text, skipping scripts, styles, nav, headers and footers."""

    _skip = {"script", "style", "nav", "header", "footer", "noscript", "svg", "math", "head"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._skip:
            self._depth += 1
        elif tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "section"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._skip and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if not self._depth:
            self.parts.append(data)


def html_to_text(raw_html: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw_html)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def fetch_document_text(paper: dict[str, Any],
                        max_chars: int = config.MAX_FULLTEXT_CHARS) -> tuple[str, str]:
    """Best-effort full text for a candidate. Returns (text, how_it_was_obtained).

    arXiv papers: the arXiv HTML rendering (most recent papers have one).
    Blog posts: the post page. Falls back to title + abstract if fetching fails.
    The text is capped at `max_chars` to keep the local DB small; the cap is
    reported so it is never silent.
    """
    fallback = f"{paper.get('title', '')}\n\n{paper.get('abstract', '')}".strip()
    url = (f"https://arxiv.org/html/{paper['id']}"
           if paper.get("source_type") == "preprint" else paper.get("url"))
    if not url:
        return fallback, "abstract-only"
    try:
        text = html_to_text(_get(url, retries=1).text)
    except Exception as exc:  # noqa: BLE001
        log.info("full text unavailable for %s (%s); using abstract", paper.get("id"), exc)
        return fallback, "abstract-only"
    if len(text) < 1500:  # landing page / stub, not the document
        return fallback, "abstract-only"
    body = f"{paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}\n\n{text}"
    if len(body) > max_chars:
        return body[:max_chars], f"full-text (first {max_chars:,} chars)"
    return body, "full-text"


# =============================================================================
# Output
# =============================================================================
def write_markdown_report(content: str, output_dir: Path | str = config.OUTPUT_DIR,
                          run_date: date | None = None) -> Path:
    """Write the final report to <output_dir>/ai-research-<YYYY-MM-DD>.md."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ai-research-{(run_date or date.today()).isoformat()}.md"
    path.write_text(content, encoding="utf-8")
    return path


# =============================================================================
# Claude tool schemas for A1 (Collector)
# =============================================================================
COLLECTOR_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_arxiv",
        "description": (
            "Search arXiv (categories cs.AI, cs.CL, cs.MA, cs.LG, cs.SE) for papers whose title or "
            "abstract matches a phrase, newest first. Only papers published inside the look-back "
            "window are returned. Each call takes ~3s; use several focused phrases rather than one "
            "broad one (e.g. 'multi-agent', 'LLM agent', 'tool use', 'agent benchmark')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A word or short phrase, e.g. 'agent memory'."},
                "max_results": {"type": "integer", "description": "1-100, default 40."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_trending_papers",
        "description": (
            "Papers featured on Hugging Face Daily Papers inside the look-back window, sorted by "
            "community upvotes (a strong citation-momentum signal). Optionally filter to papers whose "
            "title/abstract/keywords contain any of the given keywords."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"},
                             "description": "Case-insensitive keyword prefixes, e.g. 'agent' also "
                                            "matches 'agentic'; empty = no filter."},
                "limit": {"type": "integer", "description": "Max papers to return, default 40."},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_lab_blog_posts",
        "description": (
            "Recent posts (inside the look-back window) from major research-lab blogs: OpenAI, "
            "Google DeepMind, Google Research, BAIR, Hugging Face, Amazon Science, LangChain, MIT News. "
            "Many posts are product or customer news; pass keywords to filter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"},
                             "description": "Case-insensitive keyword prefixes; empty = all posts."},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "check_discussion_buzz",
        "description": (
            "Measure Hacker News discussion of a candidate (by its ID) inside the window: number of "
            "stories, total points and comments. Use it on promising candidates to gauge momentum."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"candidate_id": {"type": "string"}},
            "required": ["candidate_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "store_candidates",
        "description": (
            "Save chosen candidates to the local vector database for the Processor agent. Fetches each "
            "candidate's full text, chunks and embeds it. Only IDs returned by the other tools are "
            "accepted. Can be called more than once; stops accepting once the pool limit is reached."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "note": {"type": "string",
                                     "description": "One sentence: why this is a strong candidate."},
                        },
                        "required": ["id", "note"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["candidates"],
            "additionalProperties": False,
        },
    },
]
