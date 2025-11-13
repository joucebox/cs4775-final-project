"""Initialize the test environment."""

import sys
from pathlib import Path

# Add the repository root to the Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
