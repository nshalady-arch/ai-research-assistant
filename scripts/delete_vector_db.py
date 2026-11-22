#!/usr/bin/env python3
"""Delete the vector DB directory entirely (index, chunks, metadata).

The next orchestrator run (or scripts/create_vector_db.py) recreates it.
Use this to switch embedders or to reclaim disk space.

    python scripts/delete_vector_db.py        # asks for confirmation
    python scripts/delete_vector_db.py --yes  # no prompt
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

    if not args.db_dir.exists():
        print(f"No vector DB at {args.db_dir}. Nothing to delete.")
        return 0
    if not args.yes and input(f"Permanently delete {args.db_dir}? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return 1
    VectorStore.delete(args.db_dir)
    print(f"Deleted {args.db_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
