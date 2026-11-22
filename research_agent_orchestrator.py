#!/usr/bin/env python3
"""
Research Agent Orchestrator - entry point for the AI research personal assistant.

Pipeline (sequential):

    ensure vector DB exists  ->  A1 Collector  ->  (wait)  ->  A2 Processor  ->  ai-research-<date>.md

Usage:
    python research_agent_orchestrator.py                 # full run
    python research_agent_orchestrator.py --fresh         # clear the vector DB first
    python research_agent_orchestrator.py --skip-collect  # re-run A2 on the papers already stored
    python research_agent_orchestrator.py --days 7 --top-n 5 --domain "AI safety"
    python research_agent_orchestrator.py --provider openai_compatible --model llama3.1:8b --lite
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Find and brief the most essential recent AI research.")
    ap.add_argument("--domain", default=config.DOMAIN, help=f"research domain (default: {config.DOMAIN!r})")
    ap.add_argument("--days", type=int, default=config.LOOKBACK_DAYS, help="look-back window in days")
    ap.add_argument("--top-n", type=int, default=config.TOP_N, help="maximum number of papers to select")
    ap.add_argument("--fresh", action="store_true", help="clear the vector DB before collecting")
    ap.add_argument("--skip-collect", action="store_true", help="skip A1 and process what is already stored")
    ap.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR, help="where to write the report")
    ap.add_argument("--provider", choices=["claude", "openai_compatible"], default=config.LLM_PROVIDER,
                    help=f"LLM backend (default: {config.LLM_PROVIDER})")
    ap.add_argument("--model", help="model name for the chosen provider (default: from config)")
    ap.add_argument("--lite", action="store_true",
                    help="smaller prompts and fewer calls, for small or slow local models")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return ap.parse_args(argv)


def apply_overrides(args: argparse.Namespace) -> None:
    """CLI flags override config before any agent/prompt reads it."""
    config.DOMAIN = args.domain
    config.LOOKBACK_DAYS = args.days
    config.TOP_N = args.top_n
    config.OUTPUT_DIR = args.output_dir
    config.LLM_PROVIDER = args.provider
    if args.lite:
        for name, value in config.LITE_OVERRIDES.items():
            setattr(config, name, value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "anthropic", "urllib3", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("orchestrator")
    apply_overrides(args)

    # Imported after overrides so module-level defaults see the final config.
    from collector_agent import CollectorAgent
    from llm_client import LLMError, create_llm_client
    from processor_agent import ProcessorAgent
    from vector_store import VectorStore, VectorStoreError

    started = time.monotonic()
    try:
        # 1. Make sure the vector DB exists.
        if VectorStore.exists():
            store = VectorStore.open()
            log.info("Using vector DB at %s (%s; %s)", config.VECTOR_DB_DIR, store.embedder_name, store.count())
        else:
            store = VectorStore.create()
            log.info("Created vector DB at %s (%s)", config.VECTOR_DB_DIR, store.embedder_name)
        if args.fresh:
            store.clear()
            log.info("Cleared vector DB contents (--fresh)")

        llm = create_llm_client(args.provider, args.model)
        log.info("LLM: %s / %s%s", args.provider, llm.model, " (lite mode)" if args.lite else "")

        # 2. A1 Collector - runs to completion before A2 starts.
        if args.skip_collect:
            log.info("Skipping A1 (--skip-collect)")
        else:
            a1 = CollectorAgent(llm, store, days=args.days).run()
            log.info("A1 finished: %d candidates stored (%d topped up), %d tool calls in %d turns",
                     len(a1.stored_ids), a1.topped_up, a1.tool_calls, a1.turns)
            if a1.agent_summary:
                log.info("A1 summary: %s", a1.agent_summary.replace("\n", " ")[:600])

        if store.count()["papers"] == 0:
            log.error("The vector DB is empty; nothing for A2 to process.")
            return 1

        # 3. A2 Processor - evaluates, ranks, synthesises and writes the report.
        a2 = ProcessorAgent(llm, store, days=args.days, top_n=args.top_n, output_dir=args.output_dir).run()
    except (LLMError, VectorStoreError) as exc:
        log.error("Run failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted.")
        return 130

    elapsed = time.monotonic() - started
    print()
    print(f"Report written to: {a2.report_path}")
    print(f"Selected {len(a2.selected)} of {len(a2.evaluated)} evaluated papers "
          f"in {elapsed / 60:.1f} min · LLM usage: {llm.usage}")
    for note in a2.coverage_notes:
        print(f"  note: {note.replace('**', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
