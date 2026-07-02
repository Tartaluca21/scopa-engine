"""Entry point for the Scopa GUI (Phase 6 foundation).

python gui_run.py
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Scopa GUI.")
    parser.parse_args()
    # Import lazily so `--help` does not require pygame/display initialisation.
    from gui.app import ScopaApp

    ScopaApp().run()


if __name__ == "__main__":
    main()
