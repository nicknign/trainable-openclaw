"""Sync local project files to AutoDL remote machine.

Usage:
    python scripts/autodl_sync.py                  # sync changed files
    python scripts/autodl_sync.py --full           # full sync all source files
    python scripts/autodl_sync.py --dry-run        # show what would sync
    python scripts/autodl_sync.py --exec CMD       # run command on remote
    python scripts/autodl_sync.py --tail FILE      # tail remote log file
    python scripts/autodl_sync.py --download REMOTE LOCAL  # download file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SFTP_CONFIG_PATH = PROJECT_ROOT / ".vscode" / "sftp.json"


def _load_config() -> dict:
    with open(SFTP_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _connect(cfg: dict):
    import paramiko
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(
        cfg["host"], port=cfg["port"], username=cfg["username"],
        password=cfg["password"], timeout=15,
    )
    return c


def _remote_path(cfg: dict, rel: str) -> str:
    return f"{cfg['remotePath']}/{rel}".replace("\\", "/")


def cmd_exec(command: str):
    """Execute a command on remote and print output."""
    cfg = _load_config()
    c = _connect(cfg)
    stdin, stdout, stderr = c.exec_command(command, timeout=120)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if out:
        print(out.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
    if err.strip():
        print(f"[stderr] {err[:500]}")
    c.close()


def cmd_tail(remote_file: str):
    """Tail a remote log file."""
    cfg = _load_config()
    c = _connect(cfg)
    stdin, stdout, stderr = c.exec_command(f"tail -50 {remote_file}")
    print(stdout.read().decode())
    c.close()


def cmd_download(remote_path: str, local_path: str):
    """Download a file from remote."""
    cfg = _load_config()
    c = _connect(cfg)
    sftp = c.open_sftp()
    sftp.get(remote_path, local_path)
    sftp.close()
    c.close()
    print(f"Downloaded: {remote_path} -> {local_path}")


def cmd_sync(dry_run: bool = False, full: bool = False):
    """Sync project source files to remote."""
    cfg = _load_config()

    # Files/dirs to sync
    sync_items = [
        # (local_rel_path, remote_rel_path)
        ("trainable_openclaw/evaluation", "trainable_openclaw/evaluation"),
        ("trainable_openclaw/agent/tau_bench_tools", "trainable_openclaw/agent/tau_bench_tools"),
        ("trainable_openclaw/feedback", "trainable_openclaw/feedback"),
        ("data/tau_bench/test_prompts_augmented.jsonl", "data/tau_bench/test_prompts_augmented.jsonl"),
        ("data/tau_bench/train_prompts_augmented.jsonl", "data/tau_bench/train_prompts_augmented.jsonl"),
        ("data/tau_bench/val_prompts_augmented.jsonl", "data/tau_bench/val_prompts_augmented.jsonl"),
        ("scripts/run_single_eval.py", "scripts/run_single_eval.py"),
        ("scripts/run_full_eval.py", "scripts/run_full_eval.py"),
        ("ai_scripts/remote_check_env.py", "ai_scripts/remote_check_env.py"),
        ("ai_scripts/remote_eval_one.py", "ai_scripts/remote_eval_one.py"),
    ]

    if full:
        sync_items.extend([
            ("trainable_openclaw", "trainable_openclaw"),
            ("pyproject.toml", "pyproject.toml"),
        ])

    c = _connect(cfg)
    sftp = c.open_sftp()

    for local_rel, remote_rel in sync_items:
        local = PROJECT_ROOT / local_rel
        remote = _remote_path(cfg, remote_rel)

        if not local.exists():
            print(f"  SKIP (missing): {local_rel}")
            continue

        if local.is_dir():
            _sync_dir(sftp, str(local), local_rel, remote, dry_run)
        else:
            _sync_file(sftp, str(local), remote, dry_run)

    sftp.close()
    c.close()

    if not dry_run:
        print("\nSync complete.")


def _sync_dir(sftp, local_dir: str, rel_path: str, remote_base: str, dry_run: bool):
    """Recursively sync a directory."""
    import stat as stat_mod

    # Ensure remote dir exists
    if not dry_run:
        try:
            sftp.stat(remote_base)
        except FileNotFoundError:
            _mkdir_p(sftp, remote_base)

    for entry in os.scandir(local_dir):
        if entry.name == "__pycache__":
            continue
        if entry.name.endswith(".pyc"):
            continue

        local_path = entry.path
        remote_path = f"{remote_base}/{entry.name}"

        if entry.is_dir():
            _sync_dir(sftp, local_path, f"{rel_path}/{entry.name}", remote_path, dry_run)
        else:
            _sync_file(sftp, local_path, remote_path, dry_run)


def _sync_file(sftp, local_path: str, remote_path: str, dry_run: bool):
    """Sync a single file."""
    local_size = os.path.getsize(local_path)
    local_mtime = os.path.getmtime(local_path)

    # Check remote file
    try:
        remote_stat = sftp.stat(remote_path)
        if remote_stat.st_size == local_size:
            return  # Same size, skip
    except FileNotFoundError:
        pass

    action = "UPLOAD" if not dry_run else "WOULD UPLOAD"
    size_kb = local_size / 1024
    print(f"  {action}: {local_path} ({size_kb:.1f} KB)")
    if not dry_run:
        # Ensure remote parent directory exists
        remote_dir = "/".join(remote_path.split("/")[:-1])
        _mkdir_p(sftp, remote_dir)
        sftp.put(local_path, remote_path)


def _mkdir_p(sftp, remote_dir: str):
    """Create remote directory and parents."""
    parts = remote_dir.strip("/").split("/")
    current = ""
    for part in parts:
        current += f"/{part}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def main():
    parser = argparse.ArgumentParser(description="AutoDL remote operations")
    parser.add_argument("--exec", type=str, default="", help="Execute command on remote")
    parser.add_argument("--tail", type=str, default="", help="Tail remote log file")
    parser.add_argument("--download", nargs=2, default=[], metavar=("REMOTE", "LOCAL"),
                        help="Download remote file")
    parser.add_argument("--full", action="store_true", help="Full sync (not just changed)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would sync")
    args = parser.parse_args()

    if args.exec:
        cmd_exec(args.exec)
    elif args.tail:
        cmd_tail(args.tail)
    elif args.download:
        cmd_download(args.download[0], args.download[1])
    else:
        cmd_sync(dry_run=args.dry_run, full=args.full)


if __name__ == "__main__":
    main()
