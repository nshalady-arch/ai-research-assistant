"""
A1 - Collector agent.

A Claude tool-use agent that searches several sources for recent work in the
domain, screens candidates (source authority, recency, momentum, breadth) and
stores a candidate pool in the FAISS vector store for A2.

Guardrails enforced in code, not just in the prompt:
  * Only items actually returned by a search tool can be stored (no invented
    papers, titles or URLs) - every stored record comes from the source API.
  * The look-back window is enforced by the tools.
  * The pool size is capped (config.MAX_CANDIDATES_TO_STORE).
  * If the agent stops early or stores too few items, the pool is topped up
    with the strongest items the agent was shown, ranked by objective signals.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

import config
import prompts
import tools
from llm_client import LLMClient, LLMError
from vector_store import VectorStore

log = logging.getLogger(__name__)

_VENUE_RE = re.compile(
    r"\b(accepted|to appear|camera[- ]ready|neurips|icml|iclr|acl|emnlp|naacl|aaai|ijcai|"
    r"cvpr|iccv|eccv|kdd|tmlr|jmlr|nature|science|aamas|colm)\b", re.I)


def _filter_keywords(items: list[dict[str, Any]], keywords: list[str] | None) -> list[dict[str, Any]]:
    """Keep items whose title/abstract/keywords contain a word starting with any keyword."""
    kws = [k.strip() for k in keywords or [] if k.strip()]
    if not kws:
        return items
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + ")", re.I)
    return [p for p in items
            if pattern.search(f"{p['title']} {p['abstract']} {' '.join(p.get('keywords', []))}")]


@dataclass
class CollectorResult:
    stored_ids: list[str]
    candidates_seen: int
    agent_summary: str
    topped_up: int
    turns: int
    tool_calls: int


class CollectorAgent:
    def __init__(self, llm: LLMClient, store: VectorStore, days: int = config.LOOKBACK_DAYS):
        self.llm = llm
        self.store = store
        self.days = days
        self.registry: dict[str, dict[str, Any]] = {}  # every item any tool returned
        self.shown: set[str] = set()                   # items the agent actually saw
        self.stored: list[str] = []
        self._hf_cache: list[dict[str, Any]] | None = None
        self._blog_cache: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------ run
    def run(self) -> CollectorResult:
        log.info("[A1] Collector starting (domain=%r, last %d days)", config.DOMAIN, self.days)
        summary, turns, calls = "", 0, 0
        try:
            result = self.llm.run_tool_loop(
                system=prompts.collector_system_prompt(),
                prompt=prompts.collector_user_prompt(),
                tools=tools.COLLECTOR_TOOL_SCHEMAS,
                tool_handler=self._handle_tool,
                effort=config.COLLECTOR_EFFORT,
                max_turns=config.COLLECTOR_MAX_TURNS,
                what="A1 collector",
            )
            summary, turns, calls = result.final_text, result.turns, result.tool_calls
            if result.stopped_early:
                log.warning("[A1] reached max turns (%d) before finishing", config.COLLECTOR_MAX_TURNS)
        except LLMError as exc:
            if not self.shown:
                raise
            log.error("[A1] agent loop failed (%s); continuing with what was collected", exc)
            summary = f"Agent loop ended with an error: {exc}"

        topped_up = self._top_up()
        log.info("[A1] Collector done: %d stored, %d candidates seen", len(self.stored), len(self.registry))
        return CollectorResult(self.stored, len(self.registry), summary, topped_up, turns, calls)

    # ---------------------------------------------------------- tool dispatch
    def _handle_tool(self, name: str, args: dict[str, Any]) -> str:
        handlers = {
            "search_arxiv": self._tool_search_arxiv,
            "get_trending_papers": self._tool_trending,
            "get_lab_blog_posts": self._tool_blogs,
            "check_discussion_buzz": self._tool_buzz,
            "store_candidates": self._tool_store,
        }
        if name not in handlers:
            raise ValueError(f"unknown tool {name!r}")
        return handlers[name](**args)

    def _tool_search_arxiv(self, query: str, max_results: int = config.ARXIV_MAX_RESULTS_PER_QUERY) -> str:
        papers = tools.search_arxiv(query, days=self.days, max_results=max_results)
        return self._show(papers, f"arXiv results for {query!r}")

    def _tool_trending(self, keywords: list[str] | None = None, limit: int = 40) -> str:
        if self._hf_cache is None:
            self._hf_cache = tools.fetch_hf_daily_papers(days=self.days)
        items = _filter_keywords(self._hf_cache, keywords)[: max(1, min(int(limit), 100))]
        return self._show(items, "trending papers (by HF upvotes)")

    def _tool_blogs(self, keywords: list[str] | None = None) -> str:
        if self._blog_cache is None:
            self._blog_cache = tools.fetch_lab_blog_posts(days=self.days)
        items = _filter_keywords(self._blog_cache, keywords)
        return self._show(items, "lab blog posts in window")

    def _tool_buzz(self, candidate_id: str) -> str:
        item = self.registry.get(candidate_id)
        if not item:
            raise ValueError(f"unknown candidate_id {candidate_id!r}; use an ID returned by a search tool")
        # Full titles rarely match HN posts. Use the system name before a colon (e.g.
        # "MLExpert: Towards...") or the first few words, plus the arXiv ID (HN indexes URLs).
        title = item["title"]
        name = title.split(":")[0] if ":" in title[:60] else " ".join(title.split()[:5])
        queries = [name] + ([candidate_id] if item.get("source_type") == "preprint" else [])
        buzz = tools.check_hn_discussion(queries, days=self.days)
        item["hn_stories"] = buzz["stories"]
        item["hn_points"] = buzz["total_points"]
        item["hn_comments"] = buzz["total_comments"]
        return json.dumps(buzz, ensure_ascii=False)

    def _tool_store(self, candidates: list[dict[str, str]]) -> str:
        report = []
        for c in candidates:
            cid, note = c.get("id", ""), c.get("note", "")
            report.append({"id": cid, "status": self._store_one(cid, note)})
        self.store.save()
        return json.dumps({"results": report, "stored_total": len(self.stored),
                           "pool_limit": config.MAX_CANDIDATES_TO_STORE}, ensure_ascii=False)

    # ---------------------------------------------------------------- helpers
    def _store_one(self, cid: str, note: str) -> str:
        if cid not in self.registry:
            return "rejected: unknown ID (only IDs returned by the tools can be stored)"
        if cid in self.stored:
            return "already stored"
        if len(self.stored) >= config.MAX_CANDIDATES_TO_STORE:
            return "rejected: pool limit reached"
        item = self.registry[cid]
        if not tools.in_window(item.get("published"), self.days):
            return "rejected: outside the look-back window"
        metadata = {k: v for k, v in item.items() if k != "id"}
        metadata["collector_note"] = note
        if self.store.has_paper(cid):  # kept from an earlier run: refresh signals, skip re-download
            self.store.add_paper(cid, "", metadata, autosave=False)
            self.stored.append(cid)
            return "stored (already in DB; metadata refreshed)"
        text, how = tools.fetch_document_text(item)
        metadata["text_source"] = how
        chunks = self.store.add_paper(cid, text, metadata, autosave=False)
        self.stored.append(cid)
        log.info("[A1] stored %s (%s, %d chunks): %s", cid, how, chunks, item["title"][:70])
        return f"stored ({how}, {chunks} chunks)"

    def _merge(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add items to the registry, merging signals for items seen in several sources."""
        merged = []
        for it in items:
            existing = self.registry.get(it["id"])
            if existing is None:
                self.registry[it["id"]] = dict(it)
            else:
                for key, val in it.items():
                    if key == "source":  # prefer the specific arXiv category label over the HF one
                        if "Hugging Face" in existing.get("source", "") and val and "Hugging Face" not in val:
                            existing[key] = val
                        continue
                    if val not in (None, "", [], 0) and (
                            not existing.get(key) or key in ("hf_upvotes", "hf_comments", "github_stars")):
                        existing[key] = val
            self.shown.add(it["id"])
            merged.append(self.registry[it["id"]])
        return merged

    def _show(self, items: list[dict[str, Any]], what: str) -> str:
        """Register the first TOOL_RESULT_MAX_ITEMS items and return a compact view of them:
        enough to screen, small enough to keep the model's context lean."""
        visible = self._merge(items[: config.TOOL_RESULT_MAX_ITEMS])
        header = f"{len(items)} {what}" + (
            f"; showing the first {len(visible)}" if len(items) > len(visible) else "")
        rows = []
        for p in visible:
            authors = p.get("authors", [])
            row = {
                "id": p["id"],
                "title": p["title"],
                "published": p.get("published"),
                "source": p.get("source"),
                "authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
                "abstract": (p.get("abstract") or "")[: config.TOOL_RESULT_ABSTRACT_CHARS],
            }
            for key in ("venue_comment", "hf_upvotes", "github_stars", "hn_points"):
                if p.get(key):
                    row[key] = p[key]
            rows.append(row)
        return json.dumps({"summary": header, "items": rows}, ensure_ascii=False)

    def _top_up(self) -> int:
        """Store the best remaining shown items if the agent stored too few."""
        missing = config.MIN_CANDIDATES_TO_STORE - len(self.stored)
        if missing <= 0:
            return 0
        log.info("[A1] only %d stored; topping up %d from objective signals", len(self.stored), missing)

        def signal(p: dict[str, Any]) -> float:
            return (2 * math.log1p(p.get("hf_upvotes") or 0) + math.log1p(p.get("hn_points") or 0)
                    + (2 if _VENUE_RE.search(p.get("venue_comment") or "") else 0)
                    + (1 if p.get("source_type") == "preprint" else 0))

        pool = sorted((self.registry[i] for i in self.shown if i not in self.stored),
                      key=signal, reverse=True)
        added = 0
        for p in pool[:missing]:
            if self._store_one(p["id"], "Added by the collector's signal-based top-up.").startswith("stored"):
                added += 1
        self.store.save()
        return added
