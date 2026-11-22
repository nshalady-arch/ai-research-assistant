#!/usr/bin/env python3
"""Create the local FAISS vector DB if it does not already exist.

    python scripts/create_vector_db.py                    # embedder from config (default: auto)
    python scripts/create_vector_db.py --embedder hashing # force the offline hashing embedder
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from vector_store import VectorStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db-dir", type=Path, default=config.VECTOR_DB_DIR)
    ap.add_argument("--embedder", default=config.EMBEDDER,
                    help="auto | hashing | hashing-<dim> | st:<sentence-transformers model>")
    args = ap.parse_args()

    if VectorStore.exists(args.db_dir):
        store = VectorStore.open(args.db_dir)
        print(f"Vector DB already exists at {args.db_dir} ({store.embedder_name}; {store.count()}). Nothing to do.")
        return 0
    store = VectorStore.create(args.db_dir, embedder_spec=args.embedder)
    print(f"Created empty vector DB at {args.db_dir} (embedder: {store.embedder_name}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
