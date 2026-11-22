#!/usr/bin/env python3
"""Remove all stored papers and chunks from the vector DB, keeping the DB itself.

Use this between runs when you want the next run to start from an empty pool.

    python scripts/clear_vector_db.py        # asks for confirmation
    python scripts/clear_vector_db.py --yes  # no prompt
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
    ap.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    if not VectorStore.exists(args.db_dir):
        print(f"No vector DB at {args.db_dir}. Create one with: python scripts/create_vector_db.py")
        return 1
    store = VectorStore.open(args.db_dir)
    counts = store.count()
    if counts["papers"] == 0:
        print("Vector DB is already empty.")
        return 0
    if not args.yes and input(f"Clear {counts['papers']} papers / {counts['chunks']} chunks from "
                              f"{args.db_dir}? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return 1
    store.clear()
    print(f"Cleared {counts['papers']} papers / {counts['chunks']} chunks. The DB is empty and ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
