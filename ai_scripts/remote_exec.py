"""Execute commands on remote Linux GPU machine via paramiko SSH.

Usage:
    python ai_scripts/remote_exec.py "cmd1" "cmd2" ...
    python ai_scripts/remote_exec.py --check   # run standard health check
"""

import sys
import time
import paramiko

HOST = "connect.westb.seetacloud.com"
PORT = 27201
USER = "root"
PASSWORD = "l5pRibOdmq4M"


def get_client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
    return c


def run_cmd(client, cmd, timeout=120):
    """Run a command and return (stdout, stderr, exit_code)."""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    return stdout.read().decode("utf-8", errors="replace"), stderr.read().decode("utf-8", errors="replace"), stdout.channel.recv_exit_status()


def run_cmd_stream(client, cmd, timeout=120):
    """Run a command and stream output line by line. Returns full output."""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    output = []
    for line in iter(stdout.readline, ""):
        print(line, end="")
        output.append(line)
    err = stderr.read().decode("utf-8", errors="replace")
    exit_code = stdout.channel.recv_exit_status()
    return "".join(output), err, exit_code


def run_cmd_bg(client, cmd):
    """Run a command in background via nohup. Returns immediately."""
    full_cmd = f"nohup bash -c '{cmd}' > /tmp/bg_cmd.log 2>&1 & echo PID=$!"
    return run_cmd(client, full_cmd)


def health_check():
    """Standard remote health check."""
    client = get_client()
    try:
        checks = [
            ("GPU Info", "nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader"),
            ("Port 8000 (vllm)", "ss -tlnp | grep 8000 || echo 'NOT LISTENING'"),
            ("Port 8900 (nanobot)", "ss -tlnp | grep 8900 || echo 'NOT LISTENING'"),
            ("Python version", "/data/anaconda3/bin/python --version"),
            ("vllm installed?", "/data/anaconda3/bin/python -c 'import vllm; print(vllm.__version__)' 2>&1 || echo 'vllm NOT installed'"),
            ("openai installed?", "/data/anaconda3/bin/python -c 'import openai; print(openai.__version__)' 2>&1 || echo 'openai NOT installed'"),
            (".env file", "cat /data/wangye/trainable-openclaw/.env 2>/dev/null || echo 'NOT FOUND'"),
            ("Qwen3-4B model files", "ls /data/models/Qwen3-4B/ | head -15"),
            ("Data file (first task)", "head -1 /data/wangye/trainable-openclaw/data/tau_bench/test_prompts_augmented.jsonl | /data/anaconda3/bin/python -c \"import json,sys; d=json.load(sys.stdin); print('id:', d.get('id')); print('domain:', d.get('domain')); print('tools:', d.get('tools')); print('prompt[:200]:', d.get('prompt','')[:200])\""),
            ("nanobot api dir", "ls /data/wangye/trainable-openclaw/nanobot-0.2.1/nanobot/ 2>/dev/null || echo 'NOT FOUND'"),
            ("nanobot serve args", "grep -r 'port\\|add_argument.*port' /data/wangye/trainable-openclaw/nanobot-0.2.1/nanobot/ --include='*.py' -l 2>/dev/null | head -5 || echo 'NOT FOUND'"),
            ("Memory usage", "free -h | head -2"),
            ("Disk space", "df -h /data | tail -1"),
            ("vllm process", "ps aux | grep vllm | grep -v grep || echo 'No vllm process'"),
            ("nanobot process", "ps aux | grep nanobot | grep -v grep || echo 'No nanobot process'"),
        ]
        for name, cmd in checks:
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            out, err, rc = run_cmd(client, cmd, timeout=30)
            if out.strip():
                print(out.strip())
            if err.strip():
                print(f"  [stderr] {err.strip()[:200]}")
    finally:
        client.close()


def main():
    if len(sys.argv) == 1 or "--check" in sys.argv:
        health_check()
    else:
        client = get_client()
        try:
            for cmd in sys.argv[1:]:
                print(f"\n$ {cmd}")
                out, err, rc = run_cmd(client, cmd)
                print(out.strip())
                if err.strip():
                    print(f"[stderr] {err.strip()[:200]}")
        finally:
            client.close()


if __name__ == "__main__":
    main()
