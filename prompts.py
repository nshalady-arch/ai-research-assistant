"""
The original research-agent prompt, broken into reusable parts.

The source prompt (see README.md) is one monolithic instruction. Here it is
split into named fragments so each agent gets only the parts it needs:

    Fragment                 A1 Collector   A2 Evaluator   A2 Brief writer
    ----------------------   ------------   ------------   ---------------
    ROLE                          x              x               x
    TASK_GOAL                     x              x               x
    CRITERIA (screening)          x
    CRITERIA (full rubric)                       x
    BREADTH_REQUIREMENT           x              x               x
    PER_PAPER_FIELDS                             x
    EXECUTIVE_BRIEF                                              x
    CONSTRAINTS                   x                              x

Fragments are templates filled with the run's domain / window / top-N, so
changing config.DOMAIN retargets every agent consistently.
"""

from __future__ import annotations

from datetime import date

import config

# --- Fragments taken from the original prompt ------------------------------------
ROLE = 'You are a research librarian and domain expert in "{domain}".'

TASK_GOAL = (
    "The overall task is to identify the {top_n} most essential articles and research papers "
    "published in the last {days} days that someone would need to read to rapidly build "
    "expert-level understanding of the current state of this field."
)

CRITERION_FOUNDATIONAL_IMPACT = (
    "Foundational impact - Does it introduce a new method, framework, or finding that shifts how "
    "practitioners think about the field?"
)
CRITERION_SOURCE_AUTHORITY = (
    "Source authority - Is it from a top-tier journal, conference, or highly cited research group? "
    "Prefer peer-reviewed or pre-print from established labs over blog posts or opinion pieces."
)
CRITERION_RECENCY_RELEVANCE = (
    "Recency & relevance - Does it address the most active debates, breakthroughs, or open problems "
    "in the field right now?"
)
CRITERION_CITATION_MOMENTUM = (
    "Citation momentum - Favor papers already generating discussion (social media buzz, early "
    "citations, conference mentions)."
)
BREADTH_REQUIREMENT = (
    "Breadth of coverage - Across all {top_n} picks, ensure a mix: at least one survey/overview "
    "piece, at least one cutting-edge empirical result, and at least one that challenges or "
    "reframes conventional wisdom."
)

PER_PAPER_FIELDS = (
    "For each selection, provide:\n"
    "- Title and author(s)\n"
    "- Source (journal, arXiv, conference, publication)\n"
    "- Publication date\n"
    "- URL (direct link)\n"
    "- Why this made the top {top_n} - 2-3 sentences on what it contributes and why it's non-skippable\n"
    "- Key takeaway - The single most important idea a reader should walk away with"
)

EXECUTIVE_BRIEF = (
    'Synthesize these into a 1-page "Executive Brief" on {brief_focus} using these {year} '
    "standards."
)

CONSTRAINTS = (
    "Constraints:\n"
    "- Strictly limit to {top_n} items - no honorable mentions.\n"
    "- All must be published within the last {days} days.\n"
    "- If fewer than {top_n} high-quality items exist from the last {days} days, say so, rather "
    "than padding the list with weaker picks."
)


def _fill(template: str) -> str:
    return template.format(
        domain=config.DOMAIN,
        days=config.LOOKBACK_DAYS,
        top_n=config.TOP_N,
        brief_focus=config.BRIEF_FOCUS,
        year=date.today().year,
    )


# --- A1: Collector ----------------------------------------------------------------
def collector_system_prompt() -> str:
    return "\n\n".join([
        _fill(ROLE),
        _fill(TASK_GOAL),
        "You are agent A1, the Collector, in a two-agent pipeline. You do NOT pick the final "
        f"{config.TOP_N}. Your job is to build a high-quality CANDIDATE POOL of "
        f"{config.MIN_CANDIDATES_TO_STORE}-{config.MAX_CANDIDATES_TO_STORE} items and store it in "
        "the vector database. Agent A2 (the Processor) will read every stored item in full, score "
        "it, and choose the final list.",
        "Screen candidates with these criteria from the selection rubric:\n"
        f"1. {CRITERION_SOURCE_AUTHORITY}\n"
        f"2. {CRITERION_RECENCY_RELEVANCE}\n"
        f"3. {CRITERION_CITATION_MOMENTUM}\n"
        f"4. {CRITERION_FOUNDATIONAL_IMPACT}",
        _fill(BREADTH_REQUIREMENT) + " Make sure the pool contains several candidates of each of "
        "those three kinds, so A2 has real choices.",
        "How to work:\n"
        "- Search several sources: run 4-8 focused arXiv searches covering the field's sub-topics, "
        "check trending papers, and check lab blog posts.\n"
        "- Only consider items that are genuinely about the domain; ignore off-topic results.\n"
        "- Use check_discussion_buzz on the most promising candidates to gauge momentum.\n"
        "- Call store_candidates with your chosen IDs (one-sentence note each). Prefer original "
        "research and surveys from established groups; include a blog post only if it is a "
        "substantive technical release from a major lab.\n"
        "- When the pool is stored, reply with a short summary of what you stored and why. "
        "Do not invent IDs; only use IDs that the tools returned.",
        _fill(
            "Hard constraint: every candidate must be published within the last {days} days "
            "(the tools already enforce this window)."
        ),
    ])


def collector_user_prompt() -> str:
    return (
        f"Build the candidate pool for the domain \"{config.DOMAIN}\" covering the last "
        f"{config.LOOKBACK_DAYS} days (today is {date.today().isoformat()}). Store between "
        f"{config.MIN_CANDIDATES_TO_STORE} and {config.MAX_CANDIDATES_TO_STORE} candidates."
    )


# --- A2: Processor - per-paper evaluation ---------------------------------------
def evaluator_system_prompt() -> str:
    return "\n\n".join([
        _fill(ROLE),
        _fill(TASK_GOAL),
        "You are agent A2, the Processor. Evaluate ONE candidate at a time, strictly and "
        "consistently, so candidates can be ranked against each other. Score each criterion "
        "from 1 (very weak) to 10 (exceptional); reserve 9-10 for work that clearly stands out "
        "this period. Base scores only on the provided text and signals.",
        "Selection criteria:\n"
        f"1. {CRITERION_FOUNDATIONAL_IMPACT}\n"
        f"2. {CRITERION_SOURCE_AUTHORITY}\n"
        f"3. {CRITERION_RECENCY_RELEVANCE}\n"
        f"4. {CRITERION_CITATION_MOMENTUM} Use the community signals provided (Hugging Face upvotes, "
        "Hacker News points, GitHub stars, venue acceptance notes); with no signal, score 3-5.",
        _fill(BREADTH_REQUIREMENT) + " To support this, classify the candidate's primary "
        "contribution type: survey (survey/overview/systematization), empirical (cutting-edge "
        "empirical result or benchmark finding), reframing (challenges or reframes conventional "
        "wisdom), method (new method/framework/system), or other.",
        "Also set is_on_topic=false if the item is not really about the domain.",
        _fill(PER_PAPER_FIELDS) + "\nYou only write the last two fields (why_top_pick, "
        "key_takeaway); title, authors, source, date and URL come from the database.",
    ])


def evaluator_user_prompt(paper: dict, excerpts: list[str]) -> str:
    signals = {
        "hugging_face_upvotes": paper.get("hf_upvotes"),
        "hugging_face_comments": paper.get("hf_comments"),
        "hacker_news_points": paper.get("hn_points"),
        "hacker_news_stories": paper.get("hn_stories"),
        "github_stars": paper.get("github_stars"),
        "venue_or_comment": paper.get("venue_comment") or None,
    }
    signal_lines = "\n".join(f"- {k}: {v}" for k, v in signals.items() if v not in (None, 0, ""))
    excerpt_text = "\n\n".join(f"[Excerpt {i + 1}]\n{t}" for i, t in enumerate(excerpts))
    return (
        f"Candidate ID: {paper['paper_id']}\n"
        f"Title: {paper.get('title')}\n"
        f"Authors: {', '.join(paper.get('authors', [])[:12]) or 'unknown'}\n"
        f"Source: {paper.get('source')}\n"
        f"Published: {paper.get('published')}\n"
        f"Collector's note: {paper.get('collector_note', '-')}\n\n"
        f"Community signals:\n{signal_lines or '- none found'}\n\n"
        f"Abstract:\n{paper.get('abstract', '')}\n\n"
        f"Retrieved excerpts from the stored document:\n{excerpt_text or '(abstract only)'}"
    )


# --- A2: Processor - Executive Brief --------------------------------------------
def brief_system_prompt() -> str:
    return "\n\n".join([
        _fill(ROLE),
        _fill(TASK_GOAL),
        "You are agent A2, the Processor, writing the final synthesis. The papers below were "
        "already selected and ranked; do not add, drop or re-rank papers.",
        _fill(EXECUTIVE_BRIEF),
        "Start your answer with a line reading exactly '**TL;DR**' followed by 3-5 one-line "
        "bullets: the most important things the reader should know this period. Then write the "
        "Executive Brief in Markdown, about one page (450-700 words): an opening "
        "paragraph on the state of the field, then the core architecture recommendations as "
        "short sections or bullets (e.g. orchestration topology, roles and delegation, memory and "
        "context, tool use, evaluation and safety, open problems). Ground every claim in the "
        "selected papers and cite them inline as [#n] using their rank number. Do not include a "
        "title heading; it is added for you. Use '###' for any section headings. Do not list the "
        "papers again; the detailed list "
        "follows the brief.",
        _fill(CONSTRAINTS),
    ])


def brief_user_prompt(selected_digest: str, excerpts: str, shortfall_note: str) -> str:
    return (
        f"Selected papers (rank, metadata, evaluation):\n{selected_digest}\n\n"
        f"Relevant excerpts retrieved from the vector database:\n{excerpts}\n\n"
        f"Coverage note from the selection step: {shortfall_note or 'none'}\n\n"
        "Write the Executive Brief now."
    )
