from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Data hook for private AVAMET dataset.")
    parser.add_argument(
        "--target",
        default="avamet_all.parquet",
        help="Expected local filename for the licensed input parquet.",
    )
    args = parser.parse_args()

    target = Path(args.target).resolve()
    print("This repository does not redistribute AVAMET raw data.")
    print("Place your licensed file at:")
    print(f"  {target}")
    print("Then run:")
    print("  python scripts/validate_input.py --input avamet_all.parquet")
    print("  python scripts/run_experiment.py --input avamet_all.parquet --workdir . --redraw-figures")


if __name__ == "__main__":
    main()

