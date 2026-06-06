"""Upload files and run eval on remote GPU server via paramiko."""
import paramiko
import os
import sys

HOST = "connect.westc.seetacloud.com"
PORT = 13738
USER = "root"
PASS = "l5pRibOdmq4M"
REMOTE_BASE = "/data/wangye/trainable-openclaw"

# Connect
print(f"Connecting to {HOST}:{PORT}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, PORT, USER, PASS)
sftp = ssh.open_sftp()

try:
    # Step 1: Upload files
    local_base = "C:/work/code/claude-code/projects/trainable-openclaw"
    files_to_upload = [
        ("data/coding_debug/rubrics_v2.json", "data/rubrics_coding_v4.json"),
        ("trainable_openclaw/training/reward_bridge.py", "trainable_openclaw/training/reward_bridge.py"),
        ("tmp/eval_testset.py", "tmp/eval_testset.py"),
    ]

    for local_rel, remote_rel in files_to_upload:
        local_path = os.path.join(local_base, local_rel).replace("\\", "/")
        remote_path = f"{REMOTE_BASE}/{remote_rel}"
        print(f"Uploading: {local_rel} -> {remote_rel}")
        # Ensure remote dir exists
        remote_dir = os.path.dirname(remote_path)
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            # Create directory recursively
            ssh.exec_command(f"mkdir -p {remote_dir}")
        sftp.put(local_path, remote_path)
        print(f"  Done ({os.path.getsize(local_path)} bytes)")

    # Step 2: Check remote state
    print("\n--- Remote GPU state ---")
    stdin, stdout, stderr = ssh.exec_command("nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader")
    print(stdout.read().decode().strip())

    print("\n--- Check processes ---")
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep -E 'serve_ppo|vllm' | grep -v grep || echo 'No serve_ppo running'")
    print(stdout.read().decode().strip())

    print("\n--- Check data files ---")
    stdin, stdout, stderr = ssh.exec_command(f"ls -la {REMOTE_BASE}/data/coding/test.jsonl {REMOTE_BASE}/data/rubrics_coding_v4.json 2>&1")
    print(stdout.read().decode().strip())

    # Step 3: Verify uploaded files
    print("\n--- Verify upload ---")
    stdin, stdout, stderr = ssh.exec_command(f"cd {REMOTE_BASE} && python3 -c \"import json; r=json.load(open('data/rubrics_coding_v4.json')); print(f'{len(r)} rubrics loaded')\"")
    print(stdout.read().decode().strip())

    print("\nAll uploads complete!")

finally:
    sftp.close()
    ssh.close()
