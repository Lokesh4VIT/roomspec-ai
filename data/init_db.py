"""Seed PostgreSQL (Neon/Supabase or local SQLite) and index Qdrant.

Usage:
    python data/init_db.py            # seed if empty, index if out of sync
    python data/init_db.py --force    # upsert seed rows and rebuild the vector index
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.db.bootstrap import bootstrap  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-upsert seed data and rebuild vectors")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(bootstrap(force_reseed=args.force, force_reindex=args.force))


if __name__ == "__main__":
    main()
