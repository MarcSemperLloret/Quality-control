from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Paper 2 QC pipeline end-to-end.")
    parser.add_argument("--input", default="avamet_all.parquet", help="Input parquet path.")
    parser.add_argument("--workdir", default=".", help="Working directory where results_v3/ will be created.")
    parser.add_argument("--budgets", default="0.05,0.10", help="Comma-separated budgets.")
    parser.add_argument("--selected-budget", type=float, default=0.10, help="Budget selected for P3 materialization.")
    parser.add_argument("--c6-enabled", choices=["true", "false"], default="true", help="Enable C6 benchmark gate.")
    parser.add_argument("--redraw-figures", action="store_true", help="Run figure redraw script after pipeline.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pipeline = repo_root / "src" / "06_run_qc_with_guarantees_v3.py"
    redraw = repo_root / "src" / "08_redraw_figures_v4_png.py"

    if not pipeline.exists():
        raise FileNotFoundError(f"Missing pipeline script: {pipeline}")

    cmd = [
        args.python,
        str(pipeline),
        "--input",
        str(args.input),
        "--workdir",
        str(args.workdir),
        "--budgets",
        str(args.budgets),
        "--c6-enabled",
        str(args.c6_enabled),
        "--selected-budget",
        str(args.selected_budget),
    ]
    _run(cmd, cwd=repo_root)

    if args.redraw_figures:
        if not redraw.exists():
            raise FileNotFoundError(f"Missing redraw script: {redraw}")
        _run([args.python, str(redraw)], cwd=repo_root)


if __name__ == "__main__":
    main()

