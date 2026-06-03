"""Upload modified files to remote GPU server and run checkpoint test."""
import paramiko
import os
import sys
import time

HOST = "connect.westc.seetacloud.com"
PORT = 13738
USER = "root"
PASS = "l5pRibOdmq4M"
REMOTE_BASE = "/data/wangye/trainable-openclaw"

LOCAL_FILES = {
    "verl-main-0516/verl/trainer/serve_ppo.py":
        f"{REMOTE_BASE}/verl-main-0516/verl/trainer/serve_ppo.py",
    "scripts/start_train.sh":
        f"{REMOTE_BASE}/scripts/start_train.sh",
    "scripts/test_checkpoint.sh":
        f"{REMOTE_BASE}/scripts/test_checkpoint.sh",
}

print("=== Connecting to remote server ===")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
print(f"Connected to {HOST}:{PORT}")

# Check GPU
stdin, stdout, stderr = client.exec_command(
    "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader"
)
gpu_info = stdout.read().decode().strip()
print(f"GPU: {gpu_info}")

# Kill any existing serve_ppo
print("\n=== Cleaning up existing processes ===")
client.exec_command("pkill -9 -f serve_ppo 2>/dev/null || true")
time.sleep(2)

# Upload files
print("\n=== Uploading files ===")
sftp = client.open_sftp()
for local_rel, remote_path in LOCAL_FILES.items():
    local_path = os.path.join(os.path.dirname(__file__), "..", local_rel)
    local_path = os.path.abspath(local_path)
    if not os.path.exists(local_path):
        print(f"  SKIP {local_rel} — not found at {local_path}")
        continue
    print(f"  Uploading {local_rel} -> {remote_path}")
    sftp.put(local_path, remote_path)
    # Make shell scripts executable
    if remote_path.endswith(".sh"):
        sftp.chmod(remote_path, 0o755)
sftp.close()
print("Upload complete.")

# Run test
print("\n=== Running checkpoint test ===")
print("(This will take ~5-8 minutes — starting serve_ppo, waiting for training, saving checkpoint)\n")

channel = client.get_transport().open_session()
channel.set_combine_stderr(True)
channel.exec_command(
    f"cd {REMOTE_BASE} && bash scripts/test_checkpoint.sh"
)

# Stream output
start = time.time()
while not channel.exit_status_ready():
    if channel.recv_ready():
        data = channel.recv(4096).decode("utf-8", errors="replace")
        sys.stdout.write(data)
        sys.stdout.flush()
    time.sleep(0.1)

# Get any remaining output
while channel.recv_ready():
    data = channel.recv(4096).decode("utf-8", errors="replace")
    sys.stdout.write(data)
    sys.stdout.flush()

exit_code = channel.recv_exit_status()
elapsed = time.time() - start
print(f"\n=== Test exit code: {exit_code} (took {elapsed:.0f}s) ===")

if exit_code == 0:
    print("CHECKPOINT TEST PASSED — ready for full training.")
else:
    print("CHECKPOINT TEST FAILED — fix issues before starting full training.")

client.close()
sys.exit(exit_code)
