"""CLI alias for court-aware player interpretation and layered preview rendering."""

from InPlay.heuristic.player_interpretation import *  # noqa: F403
from InPlay.heuristic.player_interpretation import main


if __name__ == "__main__":
    raise SystemExit(main())
