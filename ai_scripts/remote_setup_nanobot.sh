#!/bin/bash
# Run on remote to fix nanobot config and restart

set -e

# Find nanobot
if command -v nanobot &>/dev/null; then
    NANOBOT=$(command -v nanobot)
elif [ -f /root/autodl-tmp/anaconda3/bin/nanobot ]; then
    NANOBOT=/root/autodl-tmp/anaconda3/bin/nanobot
else
    # Try activating conda/pip path
    export PATH="/root/autodl-tmp/anaconda3/bin:$PATH"
    NANOBOT=$(command -v nanobot 2>/dev/null || echo "NOT_FOUND")
fi

echo "nanobot: $NANOBOT"

# Verify config
echo "=== Current config model ==="
grep '"model"' /root/.nanobot/config.json

# Kill existing
pkill -9 -f "nanobot" 2>/dev/null || true
sleep 2
echo "=== nanobot processes after kill ==="
pgrep -af nanobot || echo "(none)"

# Start fresh
echo "=== Starting nanobot ==="
nohup bash -c "PATH=/root/autodl-tmp/anaconda3/bin:\$PATH nanobot serve --config /root/.nanobot/config.json" > /tmp/nanobot_startup.log 2>&1 &
sleep 8

echo "=== nanobot log ==="
tail -20 /tmp/nanobot_startup.log

echo "=== health check ==="
curl -s http://localhost:8900/health || echo "NOT READY"

echo "=== new PID ==="
pgrep -af nanobot || echo "(none)"
