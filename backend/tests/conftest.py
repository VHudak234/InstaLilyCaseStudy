"""Shared pytest config: make the backend package importable from tests/."""

import sys
from pathlib import Path

# Add backend/ to sys.path so `import tools`, `import agent` work from tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
