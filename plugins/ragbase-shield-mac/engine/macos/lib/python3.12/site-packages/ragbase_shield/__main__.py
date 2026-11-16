"""``python -m ragbase_shield`` runs the ``ragbase-shield`` command line."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
