"""Start vllm on remote machine and wait for it to be ready."""
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
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    return c


def run_cmd(client, cmd, timeout=120):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    return out, err, rc


def run_cmd_quick(client, cmd, timeout=10):
    """Run a quick command, catching timeout."""
    try:
        return run_cmd(client, cmd, timeout=timeout)
    except Exception as e:
        return "", str(e), -1


def main():
    client = get_client()
    try:
        # Step 1: Check if vllm is already running
        print("=== Step 1: Check existing vllm ===")
        out, err, rc = run_cmd_quick(client,
            "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/v1/models 2>/dev/null || echo '000'")
        print(f"  Port 8000 status: {out.strip()}")

        # If vllm is already running, we can skip startup
        if out.strip() == "200":
            print("  vllm already running and responding!")
            models_out, _, _ = run_cmd_quick(client, "curl -s http://localhost:8000/v1/models")
            print(f"  Models: {models_out[:300]}")
        else:
            # Step 2: Kill any stuck vllm processes
            print("=== Step 2: Kill existing vllm processes ===")
            run_cmd_quick(client,
                "ps aux | grep -E 'vllm|python.*vllm' | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null; sleep 2; echo 'cleaned'")

            # Step 3: Start vllm
            print("=== Step 3: Starting vllm ===")
            start_cmd = (
                "nohup env LD_LIBRARY_PATH=/data/anaconda3/lib "
                "/data/anaconda3/bin/vllm serve /data/models/Qwen3-4B "
                "--served-model-name Qwen3-4B "
                "--port 8000 "
                "--gpu-memory-utilization 0.85 "
                "--max-model-len 12288 "
                "--dtype auto "
                "> /tmp/vllm_serve.log 2>&1 &"
            )
            out, err, rc = run_cmd_quick(client, start_cmd)
            print(f"  Launch command sent. rc={rc}")
            time.sleep(3)

            # Check log to see if it's actually starting
            out, err, rc = run_cmd_quick(client, "tail -5 /tmp/vllm_serve.log")
            print(f"  Initial log: {out.strip()[:500]}")
            if err.strip():
                print(f"  stderr: {err.strip()[:300]}")

            # Step 4: Wait for vllm to be ready
            print("\n=== Step 4: Waiting for vllm (up to 6 min) ===")
            ready = False
            for i in range(72):  # 6 minutes, checking every 5s
                time.sleep(5)
                out, err, rc = run_cmd_quick(client,
                    "curl -s http://localhost:8000/v1/models 2>/dev/null || echo 'NOT_READY'")
                elapsed = (i + 1) * 5
                if '"id"' in out and 'NOT_READY' not in out:
                    print(f"\n  vllm READY after {elapsed}s!")
                    print(f"  Response: {out[:300]}")
                    ready = True
                    break
                # Show progress
                log_tail, _, _ = run_cmd_quick(client, "tail -3 /tmp/vllm_serve.log | head -1")
                status_text = log_tail.strip()[:100] if log_tail.strip() else "loading..."
                print(f"  [{elapsed}s] {status_text}", end="        \r")

            if not ready:
                print("\n  vllm failed to start. Last 40 log lines:")
                out, err, rc = run_cmd_quick(client, "tail -40 /tmp/vllm_serve.log")
                print(out[-2000:])
                if err.strip():
                    print("stderr:", err.strip()[:500])
                sys.exit(1)

        # Step 5: Test basic chat
        print("\n=== Step 5: Test basic chat ===")
        test_body = (
            '{"model":"Qwen3-4B","messages":[{"role":"user","content":"Say hello in one word."}],'
            '"temperature":0.3,"max_tokens":50}'
        )
        out, err, rc = run_cmd(client,
            f"curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{test_body}'",
            timeout=60)
        print(f"  Chat test: {out[:300]}")
        if err.strip():
            print(f"  stderr: {err.strip()[:200]}")

        # Step 6: Test tool calling
        print("\n=== Step 6: Test tool calling ===")
        tool_body = (
            '{"model":"Qwen3-4B","messages":[{"role":"user","content":"What is 2+3? Use the calculate tool."}],'
            '"tools":[{"type":"function","function":{"name":"calculate","description":"Evaluate a math expression",'
            '"parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}],'
            '"temperature":0.0,"max_tokens":200}'
        )
        out, err, rc = run_cmd(client,
            f"curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{tool_body}'",
            timeout=60)
        print(f"  Tool test: {out[:600]}")
        if '"tool_calls"' in out:
            print("  >>> Tool calling WORKS!")
        else:
            print("  >>> No tool_calls in response. Tool calling may need setup.")
        if err.strip():
            print(f"  stderr: {err.strip()[:200]}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
