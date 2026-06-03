"""Re-run checkpoint test with updated config, then start training if pass."""
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

print("=== Connecting ===")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
print(f"Connected to {HOST}:{PORT}")

# Kill old processes + cleanup
print("\n=== Cleanup ===")
client.exec_command("pkill -9 -f serve_ppo 2>/dev/null || true")
client.exec_command("rm -rf /data/wangye/trainable-openclaw/checkpoints")
time.sleep(2)

# Upload
print("\n=== Uploading ===")
sftp = client.open_sftp()
for local_rel, remote_path in LOCAL_FILES.items():
    local_path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", local_rel
    ))
    if not os.path.exists(local_path):
        print(f"  SKIP {local_rel}")
        continue
    print(f"  {local_rel} -> {remote_path}")
    sftp.put(local_path, remote_path)
    if remote_path.endswith(".sh"):
        sftp.chmod(remote_path, 0o755)
sftp.close()
print("Upload complete.")

# Run test
print("\n=== Running checkpoint test (~6 min) ===\n")

channel = client.get_transport().open_session()
channel.set_combine_stderr(True)
channel.exec_command(f"cd {REMOTE_BASE} && bash scripts/test_checkpoint.sh")

start = time.time()
while not channel.exit_status_ready():
    if channel.recv_ready():
        data = channel.recv(4096).decode("utf-8", errors="replace")
        sys.stdout.write(data)
        sys.stdout.flush()
    time.sleep(0.1)

while channel.recv_ready():
    data = channel.recv(4096).decode("utf-8", errors="replace")
    sys.stdout.write(data)
    sys.stdout.flush()

exit_code = channel.recv_exit_status()
elapsed = time.time() - start
print(f"\n=== Test exit: {exit_code} (took {elapsed:.0f}s) ===")

if exit_code != 0:
    print("CHECKPOINT TEST FAILED — aborting.")
    client.close()
    sys.exit(1)

print("CHECKPOINT TEST PASSED — starting full training...\n")

# Start full training
stdin, stdout, stderr = client.exec_command(
    f"cd {REMOTE_BASE} && bash scripts/start_train.sh && sleep 2 && echo 'started'"
)
out = stdout.read().decode()
err = stderr.read().decode()
print(out)
if err:
    print("STDERR:", err[:500])

# Check it's running
time.sleep(3)
stdin, stdout, stderr = client.exec_command("ps aux | grep serve_ppo | grep -v grep")
ps_out = stdout.read().decode().strip()
if ps_out:
    print("Training server is running:")
    print(ps_out[:500])
else:
    print("WARNING: serve_ppo process not found — check /tmp/phase3_train.log")

client.close()
print("\nDone. Monitor: ssh root@connect.westc.seetacloud.com -p 13738")
print("  tail -f /tmp/phase3_train.log")
print("  tail -f /tmp/serve_ppo_train.log")
