"""
Local FAISS vector store shared by A1 (writer) and A2 (reader).

On-disk layout (config.VECTOR_DB_DIR, default ./vector_db/):

    manifest.json   embedder name, vector dimension, creation time
    index.faiss     FAISS IndexIDMap2(IndexFlatIP) over L2-normalised vectors
                    (inner product == cosine similarity)
    chunks.json     chunk_id -> {paper_id, chunk_index, text}
    papers.json     paper_id -> paper metadata (title, authors, url, date, ...)

A1 calls `add_paper()` for each collected document; the text is chunked,
embedded and indexed. A2 calls `list_papers()`, `get_paper_chunks()` and
`search()` (optionally restricted to specific papers) to do retrieval-augmented
evaluation and synthesis.

Embeddings: "auto" uses sentence-transformers when installed, otherwise a
built-in feature-hashing embedder (numpy only, offline, deterministic). The
embedder is recorded in the manifest so a DB is never queried with a
different embedding space than it was built with.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

import config

log = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
PAPERS_FILE = "papers.json"


class VectorStoreError(RuntimeError):
    pass


# =============================================================================
# Embedders
# =============================================================================
class HashingEmbedder:
    """Feature-hashed bag of unigrams + bigrams with sublinear TF, L2-normalised.

    Not as semantic as a neural model, but needs no download, runs anywhere and
    works well for "find the passages of this paper about X" retrieval.
    """

    _token_re = re.compile(r"[a-z0-9][a-z0-9\-]+")
    _stop = frozenset(
        "the of and to in a is for on that with as by we this are be from an or at "
        "our it which can its their these has have was were not also but into than "
        "such using use based more".split()
    )

    def __init__(self, dim: int = config.HASHING_EMBEDDER_DIM):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _tokens(self, text: str) -> list[str]:
        words = [w for w in self._token_re.findall(text.lower()) if w not in self._stop]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for tok in self._tokens(text):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                idx, sign = h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)
                counts[idx] = counts.get(idx, 0.0) + sign
            for idx, val in counts.items():
                out[row, idx] = np.sign(val) * (1.0 + np.log(abs(val))) if val else 0.0
        return _normalise(out)


class SentenceTransformerEmbedder:
    """Neural embeddings via sentence-transformers (optional dependency)."""

    def __init__(self, model_name: str = config.SENTENCE_TRANSFORMER_MODEL):
        from sentence_transformers import SentenceTransformer  # imported lazily

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st:{model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, batch_size=32, show_progress_bar=False)
        return _normalise(np.asarray(vecs, dtype=np.float32))


def make_embedder(spec: str = config.EMBEDDER):
    """Build an embedder from a spec: 'auto', 'hashing', 'hashing-<dim>' or 'st:<model>'."""
    if spec == "auto":
        try:
            import sentence_transformers  # noqa: F401

            return SentenceTransformerEmbedder()
        except ImportError:
            return HashingEmbedder()
    if spec == "hashing":
        return HashingEmbedder()
    if spec.startswith("hashing-"):
        return HashingEmbedder(int(spec.split("-", 1)[1]))
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    raise VectorStoreError(f"Unknown embedder spec: {spec!r}")


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def chunk_text(text: str, size: int = config.CHUNK_SIZE_CHARS,
               overlap: int = config.CHUNK_OVERLAP_CHARS) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start + size // 2, end), text.rfind("; ", start + size // 2, end))
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


# =============================================================================
# Vector store
# =============================================================================
class VectorStore:
    """FAISS-backed store of research papers and their text chunks."""

    def __init__(self, db_dir: Path | str = config.VECTOR_DB_DIR):
        self.db_dir = Path(db_dir)
        self._embedder = None
        self._index: faiss.Index | None = None
        self._chunks: dict[int, dict[str, Any]] = {}
        self._papers: dict[str, dict[str, Any]] = {}
        self._manifest: dict[str, Any] = {}

    # ---------------------------------------------------------------- lifecycle
    @staticmethod
    def exists(db_dir: Path | str = config.VECTOR_DB_DIR) -> bool:
        d = Path(db_dir)
        return (d / MANIFEST_FILE).exists() and (d / INDEX_FILE).exists()

    @classmethod
    def create(cls, db_dir: Path | str = config.VECTOR_DB_DIR,
               embedder_spec: str = config.EMBEDDER) -> "VectorStore":
        """Create an empty DB on disk. Fails if one already exists."""
        store = cls(db_dir)
        if cls.exists(store.db_dir):
            raise VectorStoreError(f"A vector DB already exists at {store.db_dir}")
        store.db_dir.mkdir(parents=True, exist_ok=True)
        store._embedder = make_embedder(embedder_spec)
        store._manifest = {
            "embedder": store._embedder.name,
            "dim": store._embedder.dim,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "next_chunk_id": 0,
        }
        store._index = _new_index(store._embedder.dim)
        store.save()
        return store

    @classmethod
    def open(cls, db_dir: Path | str = config.VECTOR_DB_DIR) -> "VectorStore":
        """Load an existing DB from disk."""
        store = cls(db_dir)
        if not cls.exists(store.db_dir):
            raise VectorStoreError(
                f"No vector DB at {store.db_dir}. Run: python scripts/create_vector_db.py"
            )
        store._manifest = json.loads((store.db_dir / MANIFEST_FILE).read_text())
        store._index = faiss.read_index(str(store.db_dir / INDEX_FILE))
        store._chunks = {int(k): v for k, v in _read_json(store.db_dir / CHUNKS_FILE, {}).items()}
        store._papers = _read_json(store.db_dir / PAPERS_FILE, {})
        store._embedder = make_embedder(store._manifest["embedder"])  # names are valid specs
        if store._embedder.dim != store._manifest["dim"]:
            raise VectorStoreError("Embedder dimension does not match the stored index; recreate the DB.")
        return store

    @classmethod
    def open_or_create(cls, db_dir: Path | str = config.VECTOR_DB_DIR) -> "VectorStore":
        return cls.open(db_dir) if cls.exists(db_dir) else cls.create(db_dir)

    def save(self) -> None:
        self.db_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.db_dir / INDEX_FILE))
        _write_json(self.db_dir / CHUNKS_FILE, {str(k): v for k, v in self._chunks.items()})
        _write_json(self.db_dir / PAPERS_FILE, self._papers)
        _write_json(self.db_dir / MANIFEST_FILE, self._manifest)

    def clear(self) -> None:
        """Remove all papers/chunks but keep the DB (and its embedder choice)."""
        self._index = _new_index(self._manifest["dim"])
        self._chunks, self._papers = {}, {}
        self._manifest["next_chunk_id"] = 0
        self._manifest["cleared_at"] = datetime.now(timezone.utc).isoformat()
        self.save()

    @staticmethod
    def delete(db_dir: Path | str = config.VECTOR_DB_DIR) -> bool:
        """Delete the DB directory entirely. Returns False if nothing was there."""
        d = Path(db_dir)
        if not d.exists():
            return False
        if not (d / MANIFEST_FILE).exists():
            raise VectorStoreError(f"{d} does not look like a vector DB (no {MANIFEST_FILE}); refusing to delete.")
        shutil.rmtree(d)
        return True

    # ------------------------------------------------------------------- writes
    def has_paper(self, paper_id: str) -> bool:
        return paper_id in self._papers

    def add_paper(self, paper_id: str, text: str, metadata: dict[str, Any],
                  autosave: bool = True) -> int:
        """Chunk, embed and index one document. Returns the number of chunks added.

        Re-adding an existing paper_id replaces its metadata but keeps its chunks
        (the first full-text fetch wins), so repeated runs don't duplicate content.
        """
        if paper_id in self._papers:
            self._papers[paper_id].update(metadata)
            if autosave:
                self.save()
            return 0
        chunks = chunk_text(text) or chunk_text(metadata.get("abstract", "")) or [metadata.get("title", paper_id)]
        start = int(self._manifest["next_chunk_id"])
        ids = np.arange(start, start + len(chunks), dtype=np.int64)
        self._index.add_with_ids(self._embedder.embed(chunks), ids)
        for i, (cid, chunk) in enumerate(zip(ids.tolist(), chunks)):
            self._chunks[cid] = {"paper_id": paper_id, "chunk_index": i, "text": chunk}
        self._papers[paper_id] = {
            **metadata,
            "paper_id": paper_id,
            "chunk_ids": ids.tolist(),
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        self._manifest["next_chunk_id"] = start + len(chunks)
        if autosave:
            self.save()
        return len(chunks)

    # -------------------------------------------------------------------- reads
    def count(self) -> dict[str, int]:
        return {"papers": len(self._papers), "chunks": int(self._index.ntotal)}

    def list_papers(self) -> list[dict[str, Any]]:
        return [dict(p) for p in self._papers.values()]

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        p = self._papers.get(paper_id)
        return dict(p) if p else None

    def get_paper_chunks(self, paper_id: str) -> list[str]:
        p = self._papers.get(paper_id)
        return [self._chunks[c]["text"] for c in p["chunk_ids"]] if p else []

    def search(self, query: str, k: int = 5,
               paper_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Top-k chunks by cosine similarity, optionally restricted to some papers."""
        if self._index.ntotal == 0:
            return []
        q = self._embedder.embed([query])
        if paper_ids is None:
            scores, ids = self._index.search(q, min(k, self._index.ntotal))
            pairs = zip(ids[0].tolist(), scores[0].tolist())
        else:
            # Exact search over just the requested papers' vectors.
            cids = [c for pid in paper_ids for c in self._papers.get(pid, {}).get("chunk_ids", [])]
            if not cids:
                return []
            vecs = np.vstack([self._index.reconstruct(int(c)) for c in cids])
            sims = vecs @ q[0]
            order = np.argsort(-sims)[:k]
            pairs = ((cids[i], float(sims[i])) for i in order)
        results = []
        for cid, score in pairs:
            if cid == -1 or cid not in self._chunks:
                continue
            chunk = self._chunks[cid]
            paper = self._papers.get(chunk["paper_id"], {})
            results.append({
                "paper_id": chunk["paper_id"],
                "title": paper.get("title", ""),
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "score": round(float(score), 4),
            })
        return results

    @property
    def embedder_name(self) -> str:
        return self._manifest.get("embedder", "?")


# =============================================================================
# helpers
# =============================================================================
def _new_index(dim: int) -> faiss.Index:
    # IndexIDMap2 supports reconstruct(id), used for per-paper restricted search.
    return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # atomic on POSIX: a crash never leaves a half-written file
