"""Start serve_ppo in inference-only mode, then run eval_testset.py."""
import paramiko
import time
import sys

HOST = "connect.westc.seetacloud.com"
PORT = 13738
USER = "root"
PASS = "l5pRibOdmq4M"
REMOTE_BASE = "/data/wangye/trainable-openclaw"

print("Connecting...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, PORT, USER, PASS)

def run(cmd, timeout=300):
    """Run a command and return stdout, stderr, exit_code."""
    chan = ssh.get_transport().open_session()
    chan.exec_command(cmd)
    # Read all output
    out = b""
    err = b""
    while not chan.exit_status_ready():
        if chan.recv_ready():
            out += chan.recv(4096)
        if chan.recv_stderr_ready():
            err += chan.recv_stderr(4096)
        time.sleep(0.1)
    # Drain remaining
    while chan.recv_ready():
        out += chan.recv(4096)
    while chan.recv_stderr_ready():
        err += chan.recv_stderr(4096)
    exit_code = chan.recv_exit_status()
    return out.decode('utf-8', errors='replace'), err.decode('utf-8', errors='replace'), exit_code

# Step 1: Read the start script
print("\n=== Reading startup script ===")
out, err, ec = run(f"head -100 {REMOTE_BASE}/scripts/start_train.sh")
print(out)

# Step 2: Check if serve_ppo already has the inference-only config
print("\n=== Checking for idle_timeout config ===")
out, err, ec = run(f"grep -r 'idle_timeout' {REMOTE_BASE}/scripts/start_train.sh 2>/dev/null || echo 'not found in script'")
print(out.strip())

# Step 3: Start serve_ppo in inference-only mode (background)
print("\n=== Starting serve_ppo (inference only) ===")

start_cmd = f"""cd {REMOTE_BASE} && source /data/anaconda3/etc/profile.d/conda.sh && conda activate base && nohup python -m verl.trainer.serve_ppo \\
    +actor_rollout_ref.model.path=/data/models/Qwen3-4B \\
    +actor_rollout_ref.model.trust_remote_code=true \\
    +actor_rollout_ref.rollout.name=vllm \\
    +actor_rollout_ref.rollout.mode=async \\
    +actor_rollout_ref.rollout.tensor_parallel_size=1 \\
    +actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \\
    +actor_rollout_ref.rollout.max_model_len=4096 \\
    +actor_rollout_ref.rollout.load_format=auto \\
    +actor_rollout_ref.rollout.response_length=2048 \\
    +actor_rollout_ref.hybrid_engine=false \\
    +actor_rollout_ref.lora_rank=0 \\
    +trainer.total_epochs=1 \\
    +trainer.logger='[console]' \\
    +trainer.idle_timeout=999999 \\
    +trainer.min_samples=999999 \\
    +trainer.gsm8k.enabled=false \\
    +trainer.rubrics_path=data/rubrics_coding_v4.json \\
    > /tmp/serve_ppo_inference.log 2>&1 &
    echo "PID: $!"
"""

out, err, ec = run(start_cmd)
print(f"Start result: {out.strip()}")
if err:
    print(f"Stderr: {err[:500]}")

# Step 4: Wait for server to be ready
print("\n=== Waiting for server to start (may take 30-60s) ===")
for i in range(60):
    time.sleep(3)
    out, err, ec = run("curl -s http://localhost:8000/v1/health 2>/dev/null || echo 'NOT_READY'")
    if "NOT_READY" not in out and "ok" in out.lower():
        print(f"Server ready after {i*3}s!")
        print(f"Health response: {out.strip()}")
        break
    if "NOT_READY" not in out and out.strip():
        print(f"Response at {i*3}s: {out.strip()[:100]}")
    if i % 10 == 0 and i > 0:
        print(f"  Still waiting... ({i*3}s)")
else:
    print("Server may not be ready yet, proceeding anyway...")

# Step 5: Check server log
print("\n=== Recent server log ===")
out, err, ec = run(f"tail -20 /tmp/serve_ppo_inference.log")
print(out[-2000:] if len(out) > 2000 else out)

ssh.close()
print("\nDone! Check /tmp/serve_ppo_inference.log on remote for full output.")
