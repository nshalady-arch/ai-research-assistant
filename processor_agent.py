"""
A2 - Processor agent.

Reads the candidate pool A1 stored in the FAISS vector store and:

  1. Enforces the recency constraint (published within the look-back window).
  2. Evaluates every candidate with Claude against the rubric from the research
     prompt, using retrieval (RAG) over that paper's stored chunks as evidence.
  3. Ranks candidates by a weighted score, drops anything below the quality bar
     (no padding), and enforces breadth of coverage (survey / empirical /
     reframing) when qualifying candidates of those types exist.
  4. Synthesises a TL;DR + 1-page Executive Brief from the selected papers and
     excerpts retrieved from the vector store.
  5. Renders the Markdown report and writes ai-research-<date>.md.

Title, authors, source, date and URL in the report always come from the
stored source metadata, never from model output, so links cannot be invented.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

import config
import prompts
import tools
from llm_client import LLMClient, LLMError
from vector_store import VectorStore

log = logging.getLogger(__name__)

ContributionType = Literal["survey", "empirical", "reframing", "method", "other"]


class PaperEvaluation(BaseModel):
    """Structured output for one candidate (schema enforced by the API)."""

    is_on_topic: bool = Field(description="False if the item is not really about the domain.")
    contribution_type: ContributionType
    foundational_impact: int = Field(description="Score 1-10.")
    source_authority: int = Field(description="Score 1-10.")
    recency_relevance: int = Field(description="Score 1-10.")
    citation_momentum: int = Field(description="Score 1-10.")
    scoring_rationale: str = Field(description="2-4 sentences justifying the four scores.")
    why_top_pick: str = Field(description="2-3 sentences: what it contributes and why it's non-skippable.")
    key_takeaway: str = Field(description="The single most important idea, one sentence.")


@dataclass
class ScoredPaper:
    paper: dict[str, Any]
    evaluation: PaperEvaluation

    @property
    def score(self) -> float:
        e = self.evaluation
        total = sum(
            weight * max(1, min(10, getattr(e, criterion)))
            for criterion, weight in config.CRITERIA_WEIGHTS.items()
        )
        return round(total, 2)

    @property
    def qualifies(self) -> bool:
        return self.evaluation.is_on_topic and self.score >= config.MIN_QUALITY_SCORE


@dataclass
class ProcessorResult:
    report_path: Path
    selected: list[ScoredPaper]
    evaluated: list[ScoredPaper]
    excluded: list[tuple[dict[str, Any], str]]
    coverage_notes: list[str] = field(default_factory=list)


class ProcessorAgent:
    def __init__(self, llm: LLMClient, store: VectorStore, days: int = config.LOOKBACK_DAYS,
                 top_n: int = config.TOP_N, output_dir: Path = config.OUTPUT_DIR):
        self.llm = llm
        self.store = store
        self.days = days
        self.top_n = top_n
        self.output_dir = output_dir

    # ------------------------------------------------------------------ run
    def run(self) -> ProcessorResult:
        papers = self.store.list_papers()
        log.info("[A2] Processor starting: %d papers in the vector DB", len(papers))

        excluded: list[tuple[dict[str, Any], str]] = []
        in_window = []
        for p in papers:
            if tools.in_window(p.get("published"), self.days):
                in_window.append(p)
            else:
                excluded.append((p, f"published {p.get('published')} - outside the {self.days}-day window"))

        evaluated = self._evaluate_all(in_window, excluded)
        for sp in evaluated:
            if not sp.evaluation.is_on_topic:
                excluded.append((sp.paper, "judged off-topic by the evaluator"))
            elif not sp.qualifies:
                excluded.append((sp.paper, f"score {sp.score} below quality bar {config.MIN_QUALITY_SCORE}"))

        selected, notes = self._select(evaluated)
        brief = self._write_brief(selected, notes) if selected else ""
        report = self._render(selected, evaluated, excluded, notes, brief, len(papers))
        path = tools.write_markdown_report(report, self.output_dir)
        log.info("[A2] Processor done: %d selected, report -> %s", len(selected), path)
        return ProcessorResult(path, selected, evaluated, excluded, notes)

    # ------------------------------------------------------------- evaluation
    def _evaluate_one(self, paper: dict[str, Any]) -> ScoredPaper:
        hits = self.store.search(
            f"{paper.get('title', '')} main contribution method key results limitations",
            k=config.EVAL_CONTEXT_CHUNKS, paper_ids=[paper["paper_id"]],
        )
        evaluation = self.llm.complete_structured(
            system=prompts.evaluator_system_prompt(),
            prompt=prompts.evaluator_user_prompt(paper, [h["text"] for h in hits]),
            schema=PaperEvaluation,
            effort=config.EVALUATOR_EFFORT,
            what=f"A2 evaluate {paper['paper_id']}",
        )
        return ScoredPaper(paper, evaluation)

    def _evaluate_all(self, papers: list[dict[str, Any]],
                      excluded: list[tuple[dict[str, Any], str]]) -> list[ScoredPaper]:
        results: list[ScoredPaper] = []
        with ThreadPoolExecutor(max_workers=max(1, config.EVAL_WORKERS)) as pool:
            futures = {pool.submit(self._evaluate_one, p): p for p in papers}
            for i, fut in enumerate(as_completed(futures), 1):
                paper = futures[fut]
                try:
                    sp = fut.result()
                    results.append(sp)
                    log.info("[A2] (%d/%d) %.2f %-9s %s", i, len(papers), sp.score,
                             sp.evaluation.contribution_type, paper.get("title", "")[:70])
                except Exception as exc:  # noqa: BLE001 - one bad paper must not sink the run
                    log.warning("[A2] evaluation failed for %s: %s", paper["paper_id"], exc)
                    excluded.append((paper, f"evaluation failed: {exc}"))
        if papers and not results:
            raise LLMError("A2: every evaluation failed; see the log above.")
        return sorted(results, key=lambda s: s.score, reverse=True)

    # -------------------------------------------------------------- selection
    def _select(self, evaluated: list[ScoredPaper]) -> tuple[list[ScoredPaper], list[str]]:
        """Top-N qualifying papers, adjusted for breadth of coverage. Never pads."""
        qualified = [s for s in evaluated if s.qualifies]  # already sorted by score
        picks = qualified[: self.top_n]
        notes: list[str] = []

        for needed in config.REQUIRED_CONTRIBUTION_TYPES:
            if any(p.evaluation.contribution_type == needed for p in picks):
                continue
            candidate = next((s for s in qualified if s not in picks
                              and s.evaluation.contribution_type == needed), None)
            if candidate is None:
                notes.append(f"No qualifying **{needed}** piece was published in the window, "
                             "so this breadth requirement could not be met without weaker picks.")
                continue
            if len(picks) < self.top_n:
                picks.append(candidate)
            else:
                # Replace the lowest-scoring pick whose type isn't needed for breadth elsewhere.
                counts: dict[str, int] = {}
                for p in picks:
                    counts[p.evaluation.contribution_type] = counts.get(p.evaluation.contribution_type, 0) + 1
                victim = next((p for p in reversed(picks)
                               if p.evaluation.contribution_type not in config.REQUIRED_CONTRIBUTION_TYPES
                               or counts[p.evaluation.contribution_type] > 1), None)
                if victim is None:
                    continue
                picks.remove(victim)
                picks.append(candidate)
                notes.append(f"Breadth of coverage: included \"{candidate.paper['title']}\" "
                             f"(best {needed}, score {candidate.score}) in place of "
                             f"\"{victim.paper['title']}\" (score {victim.score}).")
        picks.sort(key=lambda s: s.score, reverse=True)

        if len(picks) < self.top_n:
            notes.insert(0, f"Only **{len(picks)}** high-quality items (weighted score >= "
                            f"{config.MIN_QUALITY_SCORE}) were found from the last {self.days} days, "
                            f"so the list has {len(picks)} entries rather than being padded to {self.top_n}.")
        return picks, notes

    # -------------------------------------------------------------- synthesis
    def _write_brief(self, selected: list[ScoredPaper], notes: list[str]) -> str:
        digest = "\n\n".join(
            f"[#{rank}] {s.paper['title']} ({s.paper.get('source')}, {s.paper.get('published')})\n"
            f"Type: {s.evaluation.contribution_type}; score {s.score}\n"
            f"Why: {s.evaluation.why_top_pick}\nTakeaway: {s.evaluation.key_takeaway}"
            for rank, s in enumerate(selected, 1)
        )
        rank_of = {s.paper["paper_id"]: r for r, s in enumerate(selected, 1)}
        hits = self.store.search(
            f"{config.BRIEF_FOCUS} {config.DOMAIN} architecture design orchestration memory tools evaluation",
            k=config.SYNTHESIS_CONTEXT_CHUNKS, paper_ids=list(rank_of),
        )
        excerpts = "\n\n".join(f"[#{rank_of[h['paper_id']]}] {h['text']}" for h in hits)
        log.info("[A2] writing the Executive Brief from %d papers + %d excerpts", len(selected), len(hits))
        return self.llm.complete(
            system=prompts.brief_system_prompt(),
            prompt=prompts.brief_user_prompt(digest, excerpts, " ".join(notes)),
            effort=config.SYNTHESIS_EFFORT,
            what="A2 executive brief",
        )

    # -------------------------------------------------------------- rendering
    def _render(self, selected: list[ScoredPaper], evaluated: list[ScoredPaper],
                excluded: list[tuple[dict[str, Any], str]], notes: list[str],
                brief: str, pool_size: int) -> str:
        today = date.today()
        lines = [
            f"# AI Research Brief: {config.DOMAIN} ({today.isoformat()})",
            "",
            f"> Window: {tools.window_start(self.days).isoformat()} to {today.isoformat()} "
            f"(last {self.days} days) · Candidates collected: {pool_size} · Evaluated: {len(evaluated)} · "
            f"Selected: {len(selected)} of max {self.top_n} · Model: {self.llm.model}",
            "",
            f"## Executive Brief: {config.BRIEF_FOCUS[:1].upper() + config.BRIEF_FOCUS[1:]} ({today.year})",
            "",
            brief or f"_No papers met the quality bar in the last {self.days} days, so no brief was written._",
            "",
        ]
        if notes:
            lines += ["## Coverage Notes", ""] + [f"- {n}" for n in notes] + [""]

        if selected:
            lines += ["## Reading List at a Glance", "",
                      "| # | Paper | Type | Score | Key takeaway |", "|---|---|---|---|---|"]
            for rank, s in enumerate(selected, 1):
                lines.append(f"| {rank} | [{_cell(s.paper['title'])}]({s.paper['url']}) | "
                             f"{s.evaluation.contribution_type} | {s.score} | {_cell(s.evaluation.key_takeaway)} |")
            lines += ["", f"## The Top {len(selected)}: Details", ""]
            for rank, s in enumerate(selected, 1):
                p, e = s.paper, s.evaluation
                authors = p.get("authors") or ["Unknown"]
                author_text = ", ".join(authors[:8]) + (f", et al. ({len(authors)} authors)" if len(authors) > 8 else "")
                venue = f" · {p['venue_comment']}" if p.get("venue_comment") else ""
                lines += [
                    f"### {rank}. {p['title']}",
                    "",
                    f"- **Author(s):** {author_text}",
                    f"- **Source:** {p.get('source')}{venue}",
                    f"- **Publication date:** {p.get('published')}",
                    f"- **URL:** {p['url']}",
                    f"- **Type / score:** {e.contribution_type} · {s.score}/10 "
                    f"(impact {e.foundational_impact}, authority {e.source_authority}, "
                    f"relevance {e.recency_relevance}, momentum {e.citation_momentum})",
                    "",
                    f"**Why this made the top {self.top_n}:** {e.why_top_pick}",
                    "",
                    f"**Key takeaway:** {e.key_takeaway}",
                    "",
                ]

        lines += ["## Appendix: How This List Was Built", "",
                  "A1 (Collector) searched arXiv, Hugging Face Daily Papers, research-lab blogs and "
                  "Hacker News, then stored a candidate pool in the local FAISS vector DB. A2 (Processor) "
                  "scored each candidate 1-10 on foundational impact, source authority, recency & relevance "
                  "and citation momentum (weights: "
                  + ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in config.CRITERIA_WEIGHTS.items())
                  + f"), kept only on-topic items scoring at least {config.MIN_QUALITY_SCORE}, and "
                  "balanced the final list for breadth of coverage.", ""]
        if evaluated:
            lines += ["| Score | Type | Paper | Published | Selected |", "|---|---|---|---|---|"]
            chosen = {id(s) for s in selected}
            for s in evaluated:
                lines.append(f"| {s.score} | {s.evaluation.contribution_type} | "
                             f"[{_cell(s.paper['title'])}]({s.paper['url']}) | {s.paper.get('published')} | "
                             f"{'yes' if id(s) in chosen else ''} |")
            lines.append("")
        if excluded:
            lines += ["**Not selected:**", ""] + [
                f"- [{_cell(p.get('title', p.get('paper_id')))}]({p.get('url', '')}) - {reason}"
                for p, reason in excluded] + [""]
        return "\n".join(lines)


def _cell(text: str) -> str:
    """Make text safe inside a Markdown table cell or link label."""
    return str(text).replace("|", "\\|").replace("\n", " ").replace("[", "(").replace("]", ")")
