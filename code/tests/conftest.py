import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
SRC = CODE / "src"

for path in (CODE, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
