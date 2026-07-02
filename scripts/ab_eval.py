"""CLI: paired self-play A/B of the default bot vs a PIMC challenger.

The baseline is always `default_pimc_config()`. The challenger starts as a copy
of the baseline; override any lever with the flags below.

    # Smoke test (fast, low N -- not statistically meaningful):
    python scripts/ab_eval.py -n 4 --workers 4 --max-depth 6

    # Real run (README convention: N >= 150, fresh seeds, seats swapped):
    python scripts/ab_eval.py -n 150 --seed 1 --workers 8 --uniform-weights \\
        --out logs/ab_uniform_weights.md

Challenger levers: --n-worlds, --max-depth, --uniform-weights, and repeatable
--weight NAME=VALUE (captures/denari/settebello/primiera/scope).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from botconfig import default_pimc_config  # noqa: E402
from engine.features import Weights  # noqa: E402
from search.ab_eval import run_ab  # noqa: E402
from search.alphabeta import SearchConfig  # noqa: E402
from search.pimc import PimcConfig  # noqa: E402
from session import pimc_bot_name  # noqa: E402

_WEIGHT_FIELDS = ("captures", "denari", "settebello", "primiera", "scope")


def _build_challenger(args: argparse.Namespace, base: PimcConfig) -> tuple[PimcConfig, str]:
    """Derive a challenger PimcConfig from the baseline plus CLI overrides."""
    weights = Weights() if args.uniform_weights else base.search.weights
    overrides: dict[str, float] = {}
    for spec in args.weight or []:
        name, _, value = spec.partition("=")
        if name not in _WEIGHT_FIELDS or not value:
            raise SystemExit(f"bad --weight {spec!r}; use NAME=VALUE from {_WEIGHT_FIELDS}")
        overrides[name] = float(value)
    if overrides:
        weights = replace(weights, **overrides)
    n_worlds = args.n_worlds if args.n_worlds is not None else base.n_worlds
    max_depth = args.max_depth if args.max_depth is not None else base.search.max_depth
    cfg = PimcConfig(
        n_worlds=n_worlds,
        search=SearchConfig(
            max_depth=max_depth, weights=weights, use_endgame_solver=args.endgame_solver
        ),
    )
    tags = []
    if args.uniform_weights:
        tags.append("uniform-weights")
    if args.endgame_solver:
        tags.append("endgame-solver")
    if overrides:
        tags.append("+".join(f"{k}={v}" for k, v in overrides.items()))
    label = pimc_bot_name(n_worlds, max_depth) + (f" [{','.join(tags)}]" if tags else "")
    return cfg, label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n", "--pairs", type=int, default=150, help="paired deals (>=150 for power)"
    )
    parser.add_argument("--seed", type=int, default=0, help="base seed for the deck sequence")
    parser.add_argument("--workers", type=int, default=4, help="process-pool workers")
    parser.add_argument("--n-worlds", type=int, default=None, help="challenger sampled worlds")
    parser.add_argument("--max-depth", type=int, default=None, help="challenger alpha-beta depth")
    parser.add_argument("--uniform-weights", action="store_true", help="challenger uses Weights()")
    parser.add_argument(
        "--endgame-solver", action="store_true", help="challenger solves deck-empty states exactly"
    )
    parser.add_argument(
        "--weight", action="append", metavar="NAME=VALUE", help="override one challenger weight"
    )
    parser.add_argument("--out", type=Path, default=None, help="also write the report to this path")
    args = parser.parse_args()

    baseline = default_pimc_config()
    base_label = pimc_bot_name(baseline.n_worlds, baseline.search.max_depth) + " [default]"
    challenger, chal_label = _build_challenger(args, baseline)

    result = run_ab(
        baseline,
        challenger,
        n_pairs=args.pairs,
        seed=args.seed,
        workers=args.workers,
        baseline_label=base_label,
        challenger_label=chal_label,
    )
    report = result.to_markdown()
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\nWrote report to {args.out}")


if __name__ == "__main__":
    main()
