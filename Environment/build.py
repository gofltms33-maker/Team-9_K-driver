"""Generate the configured SUMO road network without starting training."""
from pathlib import Path
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "generated",
                        help="Output directory (relative paths use your current working directory)")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1] / "ppo_v1_parallel"
    sys.path.insert(0, str(project))
    from env.road_config import ROAD, EGO, TRAFFIC
    from env.road_builder import build
    cfg = build(ROAD, EGO, TRAFFIC, str(args.output.resolve()))
    print(f"SUMO configuration: {cfg}")


if __name__ == "__main__":
    main()
