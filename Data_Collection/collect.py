"""Run the original collector (default: SAFE13 + B, not the final B/Y/S model)."""
from pathlib import Path
import os
import runpy
import sys


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1] / "ppo_v1_parallel"
    os.chdir(project)
    sys.path.insert(0, str(project))
    sys.argv[0] = str(project / "collect_data.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
