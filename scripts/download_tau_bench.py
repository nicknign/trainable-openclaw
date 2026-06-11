"""
Download tau-bench data from HuggingFace (APIGen) and GitHub.

Usage:
    python scripts/download_tau_bench.py          # download all
    python scripts/download_tau_bench.py --hf-only  # only HuggingFace
    python scripts/download_tau_bench.py --hf-endpoint https://hf-mirror.com  # China mirror
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "tau_bench" / "raw"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download_historical_trajectories():
    """Clone tau-bench repos and extract historical trajectories + task defs."""
    print("=" * 60)
    print("TASK 1 & 4: Downloading historical trajectories + task definitions")
    print("=" * 60)

    # The historical_trajectories/ are in the ORIGINAL tau-bench repo
    # The task JSONs are in the CURRENT tau2-bench repo at data/tau2/domains/{domain}/tasks.json
    # We'll do shallow clones with sparse checkout to minimize download size

    repos = [
        {
            "url": "https://github.com/sierra-research/tau-bench.git",
            "name": "tau-bench",
            "files": [
                "historical_trajectories/gpt-4o-airline.json",
                "historical_trajectories/gpt-4o-retail.json",
                "historical_trajectories/sonnet-35-new-airline.json",
                "historical_trajectories/sonnet-35-new-retail.json",
            ],
        },
        {
            "url": "https://github.com/sierra-research/tau2-bench.git",
            "name": "tau2-bench",
            "files": [
                "data/tau2/domains/airline/tasks.json",
                "data/tau2/domains/retail/tasks.json",
                "data/tau2/domains/airline/tools.json",
                "data/tau2/domains/retail/tools.json",
            ],
        },
    ]

    for repo in repos:
        print(f"\n--- Processing {repo['name']} ---")
        clone_dir = RAW_DIR / repo["name"]

        if (clone_dir / ".git").exists():
            print(f"  Already cloned at {clone_dir}, fetching latest...")
            subprocess.run(
                ["git", "-C", str(clone_dir), "fetch", "--depth", "1"],
                check=False,
            )
        else:
            print(f"  Shallow cloning {repo['url']} (depth=1, no checkout)...")
            result = subprocess.run(
                [
                    "git", "clone", "--depth", "1", "--filter=blob:none",
                    "--no-checkout", repo["url"], str(clone_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"  Clone failed: {result.stderr}")
                print(f"  Trying full clone...")
                subprocess.run(
                    ["git", "clone", "--depth", "1", repo["url"], str(clone_dir)],
                    check=False,
                )

        # Sparse checkout the specific files we need
        if (clone_dir / ".git").exists():
            print(f"  Configuring sparse checkout...")
            subprocess.run(
                ["git", "-C", str(clone_dir), "config", "core.sparseCheckout", "true"],
                check=False,
            )
            sparse_file = clone_dir / ".git" / "info" / "sparse-checkout"
            ensure_dir(sparse_file.parent)
            with open(sparse_file, "w") as f:
                for filepath in repo["files"]:
                    f.write(filepath + "\n")
            subprocess.run(
                ["git", "-C", str(clone_dir), "checkout"],
                check=False,
            )

        # Copy files to raw directory
        for filepath in repo["files"]:
            src = clone_dir / filepath
            dst = RAW_DIR / Path(filepath).name
            if src.exists():
                import shutil
                shutil.copy2(str(src), str(dst))
                size_kb = src.stat().st_size / 1024
                print(f"  Copied: {filepath} -> raw/{dst.name} ({size_kb:.1f} KB)")
            else:
                print(f"  MISSING: {filepath} not found in checkout")


def download_apigen_hf(endpoint=None):
    """Download APIGen tau-bench data from HuggingFace."""
    print("\n" + "=" * 60)
    print("TASK 2: Downloading APIGen tau-bench from HuggingFace")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets library not installed. Install with: pip install datasets")
        return

    kwargs = {"path": "amityco/apigen-tau-bench-split-turn", "split": "train", "trust_remote_code": True}
    if endpoint:
        kwargs["download_config"] = type("D", (), {"endpoint": endpoint})()
        # Actually HF uses HF_ENDPOINT env var - set it
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"  Using HF endpoint: {endpoint}")

    print(f"  Loading dataset: {kwargs['path']} ...")
    try:
        dataset = load_dataset(**kwargs)
    except Exception as e:
        print(f"  Loading with kwargs failed: {e}")
        print(f"  Trying without trust_remote_code...")
        kwargs.pop("trust_remote_code")
        dataset = load_dataset(**kwargs)

    print(f"  Dataset loaded: {len(dataset)} rows")
    print(f"  Columns: {dataset.column_names}")
    print(f"  Features: {dataset.features}")

    # Save first 100 samples for format analysis
    sample_path = RAW_DIR / "apigen_sample.json"
    samples = dataset[:100]
    # Convert to serializable format
    serializable = []
    for i in range(min(100, len(dataset))):
        row = {}
        for col in dataset.column_names:
            row[col] = dataset[i][col]
        serializable.append(row)

    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)

    print(f"  Saved {len(serializable)} samples to {sample_path}")
    print(f"  Sample file size: {sample_path.stat().st_size / 1024:.1f} KB")

    # Also save a single row for detailed inspection
    single_path = RAW_DIR / "apigen_single_sample.json"
    if len(dataset) > 0:
        row = {}
        for col in dataset.column_names:
            row[col] = dataset[0][col]
        with open(single_path, "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Saved single sample to {single_path}")


def main():
    parser = argparse.ArgumentParser(description="Download tau-bench data")
    parser.add_argument("--hf-only", action="store_true", help="Only download HuggingFace data")
    parser.add_argument("--traj-only", action="store_true", help="Only download historical trajectories")
    parser.add_argument("--hf-endpoint", type=str, default=None,
                        help="HF endpoint (e.g., https://hf-mirror.com)")
    args = parser.parse_args()

    ensure_dir(RAW_DIR)

    if args.hf_only:
        download_apigen_hf(args.hf_endpoint)
    elif args.traj_only:
        download_historical_trajectories()
    else:
        download_historical_trajectories()
        download_apigen_hf(args.hf_endpoint)

    print(f"\nDone. Data saved to: {RAW_DIR}")


if __name__ == "__main__":
    main()
