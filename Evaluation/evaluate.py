"""Evaluate PPO; default to the final 7:3 checkpoint with B/Y/S observations."""
from pathlib import Path
import os
import runpy
import sys


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1] / "ppo_v1_parallel"
    os.chdir(project)
    sys.path.insert(0, str(project))
    args = sys.argv[1:]
    # A custom checkpoint, when supplied, must be the first argument.
    if not args or args[0].startswith("-"):
        args.insert(0, "results/MIX13x17_w70/model.pt")
    for option, value in (("--yield-cut", "on"), ("--yellow-obs", "on"),
                          ("--stalled-block", "on"), ("--warmup", "300-600")):
        if not any(arg == option or arg.startswith(option + "=") for arg in args):
            args.extend((option, value))
    sys.argv = [str(project / "test.py"), *args]
    runpy.run_path(sys.argv[0], run_name="__main__")
