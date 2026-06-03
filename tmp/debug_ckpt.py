"""Debug: check checkpoint files on remote server."""
import paramiko
import sys

HOST = "connect.westc.seetacloud.com"
PORT = 13738
USER = "root"
PASS = "l5pRibOdmq4M"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

# Full file listing
stdin, stdout, stderr = client.exec_command(
    "find /tmp/ckpt_test -type f -o -type d | sort 2>/dev/null; "
    "echo '---'; "
    "ls -laR /tmp/ckpt_test/ 2>/dev/null"
)
print(stdout.read().decode())

# Check the training log for any checkpoint-related errors
stdin, stdout, stderr = client.exec_command(
    "grep -i 'checkpoint\\|save\\|optim\\|extra\\|huggingface\\|rank\\|world_size' /tmp/ckpt_test_train.log 2>/dev/null | head -40"
)
print("\n=== Log grep ===")
print(stdout.read().decode())

client.close()
