# -*- coding: utf-8 -*-
"""
main.py

Master script for Assignment 3 GKX replication.

Runs all project steps in order:

    01_build_panel.py
    02_train_simple_models.py
    03_train_ml_models.py
    04_make_figure4.py
    05_make_figure9.py
    06_make_final1_table.py

Useful options:

    python run_all.py --from-step 4
    python run_all.py --to-step 3
    python run_all.py --skip-data
    python run_all.py --skip-training
"""

from pathlib import Path
import argparse
import subprocess
import sys
import time


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
OUTPUT_DIR = PROJECT_DIR / "outputs"
LOG_DIR = OUTPUT_DIR / "run_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


STEPS = [
    {
        "step": 1,
        "name": "Build panel",
        "script": SRC_DIR / "01_build_panel.py",
        "kind": "data",
    },
    {
        "step": 2,
        "name": "Train simple models",
        "script": SRC_DIR / "02_train_simple_models.py",
        "kind": "training",
    },
    {
        "step": 3,
        "name": "Train ML models",
        "script": SRC_DIR / "03_train_ml_models.py",
        "kind": "training",
    },
    {
        "step": 4,
        "name": "Make Figure 4",
        "script": SRC_DIR / "04_make_figure4.py",
        "kind": "figures",
    },
    {
        "step": 5,
        "name": "Make Figure 9",
        "script": SRC_DIR / "05_make_figure9.py",
        "kind": "figures",
    },
    {
        "step": 6,
        "name": "Make final Table 1",
        "script": SRC_DIR / "06_make_final1_table.py",
        "kind": "table",
    },
]


def run_step(step_info):
    step = step_info["step"]
    name = step_info["name"]
    script = step_info["script"]

    if not script.exists():
        raise FileNotFoundError(
            f"Step {step} script not found:\n{script}\n\n"
            "Check that your file is in src/ and that the filename does not contain '(1)'."
        )

    print("=" * 90)
    print(f"[STEP {step}] {name}")
    print(f"Script: {script}")
    print("=" * 90)

    start = time.time()

    log_path = LOG_DIR / f"step{step}_{script.stem}.log"

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return_code = process.wait()

    elapsed = time.time() - start

    if return_code != 0:
        raise RuntimeError(
            f"\nStep {step} failed with exit code {return_code}.\n"
            f"Check log file:\n{log_path}"
        )

    print("-" * 90)
    print(f"[STEP {step} COMPLETE] {name}")
    print(f"Elapsed time: {elapsed / 60:.2f} minutes")
    print(f"Log saved to: {log_path}")
    print("-" * 90)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all GKX assignment scripts in order."
    )

    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        help="First step to run. Default: 1.",
    )

    parser.add_argument(
        "--to-step",
        type=int,
        default=6,
        help="Last step to run. Default: 6.",
    )

    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip step 1, panel construction.",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip steps 2 and 3, model training.",
    )

    parser.add_argument(
        "--only-figures",
        action="store_true",
        help="Run only steps 4 and 5.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 90)
    print("GKX Assignment 3 master run")
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Python executable: {sys.executable}")
    print("=" * 90)

    selected_steps = []

    for s in STEPS:
        step_num = s["step"]

        if step_num < args.from_step or step_num > args.to_step:
            continue

        if args.skip_data and s["kind"] == "data":
            continue

        if args.skip_training and s["kind"] == "training":
            continue

        if args.only_figures and s["kind"] != "figures":
            continue

        selected_steps.append(s)

    if not selected_steps:
        print("No steps selected. Nothing to run.")
        return

    print("Selected steps:")
    for s in selected_steps:
        print(f"  Step {s['step']}: {s['name']}")

    print("=" * 90)

    overall_start = time.time()

    for s in selected_steps:
        run_step(s)

    overall_elapsed = time.time() - overall_start

    print("=" * 90)
    print("ALL SELECTED STEPS COMPLETE")
    print(f"Total elapsed time: {overall_elapsed / 60:.2f} minutes")
    print("=" * 90)


if __name__ == "__main__":
    main()