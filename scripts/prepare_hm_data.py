"""Build the default stratified-customer local CSV runtime cache."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import DataAnalyzer
from src.runtime import discover_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    env = {"HM_RAW_DATA_DIR": str(args.raw_dir)}
    if args.runtime_dir:
        env["HM_RUNTIME_DIR"] = str(args.runtime_dir)
    analyzer = DataAnalyzer(discover_runtime(Path.cwd(), env), args.chunksize)
    print(json.dumps(analyzer.load_data(), indent=2))


if __name__ == "__main__":
    main()
