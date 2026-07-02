"""Report the human's record vs the bot over all logged deals.

Reads the JSONL written by `play.py` (via `gamelog`) and prints win-rate,
margins, scope, and per-component win rates. Robust to mixed logs: legacy rows
without the rich fields simply drop out of the metrics that need them, and each
such metric prints the count of deals it was actually computed from. Run with:

    python stats.py
"""

from __future__ import annotations

from gamelog import DealRecord, read_deals

_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("Settebello", "settebello_winner"),
    ("Denari", "denari_winner"),
    ("Primiera", "primiera_winner"),
    ("Cards", "cards_winner"),
)


def _win_rate(wins: int, ties: int, total: int) -> float:
    """Win-rate counting ties as half a win (the standard convention)."""
    return (wins + 0.5 * ties) / total


def _component_line(label: str, field: str, deals: list[DealRecord]) -> str:
    present = [getattr(d, field) for d in deals if getattr(d, field) is not None]
    if not present:
        return f"  {label:<11}: n/a  (no rich rows)"
    wins = present.count("human")
    ties = present.count("none")
    rate = _win_rate(wins, ties, len(present))
    return f"  {label:<11}: {rate:5.1%}  (over {len(present)} deals)"


def _avg_scope_line(deals: list[DealRecord]) -> str:
    rich = [d for d in deals if d.human_scope is not None and d.bot_scope is not None]
    if not rich:
        return "Avg scope    : n/a  (no rich rows)"
    human = sum(d.human_scope for d in rich if d.human_scope is not None) / len(rich)
    bot = sum(d.bot_scope for d in rich if d.bot_scope is not None) / len(rich)
    return f"Avg scope    : you {human:.2f}  vs  bot {bot:.2f}  (over {len(rich)} deals)"


def summarize(deals: list[DealRecord]) -> str:
    """Format the full win-rate, margin, scope, and component report."""
    n = len(deals)
    if n == 0:
        return "No deals logged yet. Play a game with `python play.py` first."
    wins = sum(1 for d in deals if d.result == "win")
    losses = sum(1 for d in deals if d.result == "loss")
    ties = sum(1 for d in deals if d.result == "tie")
    lines = [
        f"Deals played : {n}",
        f"Record       : {wins}W - {losses}L - {ties}T",
        f"Win-rate     : {_win_rate(wins, ties, n):.1%}  (ties as half)",
        f"Avg margin   : {sum(d.margin for d in deals) / n:+.2f}  (you - bot)",
        f"Avg score    : you {sum(d.human for d in deals) / n:.2f}"
        f"  vs  bot {sum(d.bot for d in deals) / n:.2f}",
        _avg_scope_line(deals),
        "Component win rates (ties as half):",
    ]
    lines.extend(_component_line(label, field, deals) for label, field in _COMPONENTS)
    return "\n".join(lines)


def main() -> None:
    print(summarize(read_deals()))


if __name__ == "__main__":
    main()
