"""Run main.py against a brand with MongoDB stubbed out.

Patches pymongo.MongoClient with tests.fixtures.mongo_stub.StubMongoClient
BEFORE importing main, sets sys.argv from this script's argv, and invokes
main.main(). Intended for one-off PDF generation when no real Mongo is
available (CI smoke runs, sandbox demos, dev laptops without docker).

Run:
    GEMINI_API_KEY=... BRAND_DNA_OUTPUT_DIR=output/v2 \\
        python3 tests/fixtures/run_mocked.py \\
            --config=local --name=kaft \\
            --url=https://www.kaft.com/ --social_handle=@kaft
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _patch_mongo() -> None:
    import pymongo

    from tests.fixtures.mongo_stub import StubMongoClient

    pymongo.MongoClient = StubMongoClient


def main() -> None:
    _patch_mongo()
    # main.py reads sys.argv at import time via argparse, so do NOT pop our
    # own script name — argparse skips argv[0]. Just pass the brand args
    # exactly as you would to `python3 main.py ...`.
    import main as agent_main

    agent_main.main()


if __name__ == "__main__":
    main()
