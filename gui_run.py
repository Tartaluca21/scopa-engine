"""Entry point for the Scopa GUI (Phase 6 foundation).

python gui_run.py
"""

from __future__ import annotations

from gui.app import ScopaApp


def main() -> None:
    ScopaApp().run()


if __name__ == "__main__":
    main()
