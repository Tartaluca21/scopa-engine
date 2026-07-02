"""CLI: build a human-decision dataset from captured deal logs.

    python scripts/build_decision_dataset.py            # write dataset + report
    python scripts/build_decision_dataset.py --report   # report only, no write
    python scripts/build_decision_dataset.py --sample 5 # also pretty-print 5 rows

Requires deals captured via `python play.py --record-moves`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly (`python scripts/...`) by putting the
# project root on the path so the top-level modules import cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decision_dataset import (  # noqa: E402
    DEFAULT_OUTPUT,
    human_decision_rows,
    report_text,
    sample_text,
    write_jsonl,
)
from gamelog import LOG_PATH, MATCH_LOG_PATH, read_deals, read_matches  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=LOG_PATH, help="deal log (JSONL)")
    parser.add_argument("--matches", type=Path, default=MATCH_LOG_PATH, help="match log (JSONL)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="dataset output path")
    parser.add_argument("--report", action="store_true", help="print report only, do not write")
    parser.add_argument(
        "--sample", type=int, default=0, metavar="N", help="pretty-print the first N decisions"
    )
    args = parser.parse_args()

    deals = read_deals(args.input)
    matches = read_matches(args.matches)
    rows = human_decision_rows(deals, matches)
    if not args.report:
        write_jsonl(rows, args.output)
        print(f"Wrote {len(rows)} human decisions to {args.output}")
    print(report_text(rows, len(deals)))
    if args.sample > 0:
        print("\n" + sample_text(rows, args.sample))


if __name__ == "__main__":
    main()
