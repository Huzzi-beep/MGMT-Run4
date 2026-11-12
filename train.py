"""Launch the recovered single-slice MGMT training script."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, help="Directory containing the training datasets")
    parser.add_argument("--work-dir", type=Path, help="Directory for the training cache and outputs")
    parser.add_argument("--quick", action="store_true", help="Run the reduced training check")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.input_root is not None:
        env["MERGED_INPUT_ROOT"] = str(args.input_root.expanduser().resolve())
    if args.work_dir is not None:
        env["MERGED_WORK"] = str(args.work_dir.expanduser().resolve())
    if args.quick:
        env["MERGED_QUICK"] = "1"
    script = Path(__file__).resolve().parent / "training" / "kaggle_merged_training.py"
    subprocess.run([sys.executable, "-u", str(script)], env=env, check=True)


if __name__ == "__main__":
    main()
