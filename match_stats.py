"""Report the human's match record vs the bot over all logged matches.

Reads the JSONL written by `play.py --match-to N` (via `gamelog`) and prints
the match win-rate, margins, deal counts, and a per-bot-configuration
breakdown. Run with:

    python match_stats.py
"""

from __future__ import annotations

from gamelog import MatchRecord, read_matches


def _win_rate(wins: int, ties: int, total: int) -> float:
    """Win-rate counting ties as half a win (the standard convention)."""
    return (wins + 0.5 * ties) / total


def _record_line(matches: list[MatchRecord], indent: str = "") -> list[str]:
    n = len(matches)
    wins = sum(1 for m in matches if m.winner == "human")
    losses = sum(1 for m in matches if m.winner == "bot")
    ties = sum(1 for m in matches if m.winner == "tie")
    avg_margin = sum(m.final_margin for m in matches) / n
    avg_deals = sum(m.n_deals for m in matches) / n
    return [
        f"{indent}Matches      : {n}",
        f"{indent}Record       : {wins}W - {losses}L - {ties}T",
        f"{indent}Win-rate     : {_win_rate(wins, ties, n):.1%}  (ties as half)",
        f"{indent}Avg margin   : {avg_margin:+.2f}  (you - bot)",
        f"{indent}Avg deals    : {avg_deals:.2f}  per match",
    ]


def summarize(matches: list[MatchRecord]) -> str:
    """Format the full match report, with a per-configuration breakdown."""
    if not matches:
        return "No matches logged yet. Play one with `python play.py --match-to 11`."
    lines = _record_line(matches)
    by_bot: dict[str, list[MatchRecord]] = {}
    for m in matches:
        by_bot.setdefault(m.bot_name or "(unknown)", []).append(m)
    if len(by_bot) > 1 or "(unknown)" not in by_bot:
        lines.append("By bot configuration:")
        for name in sorted(by_bot):
            lines.append(f"  {name}:")
            lines.extend(_record_line(by_bot[name], indent="    "))
    return "\n".join(lines)


def main() -> None:
    print(summarize(read_matches()))


if __name__ == "__main__":
    main()
