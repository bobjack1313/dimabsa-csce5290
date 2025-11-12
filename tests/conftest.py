import sys
from pathlib import Path

# Add project root to sys.path so 'scripts' and 'src' are importable
sys.path.append(str(Path(__file__).resolve().parent.parent))
