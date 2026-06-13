#!/bin/bash
# Phase 4: Start nanobot backed by serve_ppo API
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export LD_LIBRARY_PATH=/data/anaconda3/lib:$LD_LIBRARY_PATH

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE="${API_BASE:-http://localhost:8000/v1}"
MODEL="${MODEL:-qwen3-4b}"
NANOBOT_PORT="${NANOBOT_PORT:-18790}"
WORKSPACE="${WORKSPACE:-$HOME/.nanobot/workspace}"

echo "=== Phase 4: nanobot + serve_ppo integration ==="
echo "  API base:   $API_BASE"
echo "  Model:      $MODEL"
echo "  Port:       $NANOBOT_PORT"
echo ""

# ---------------------------------------------------------------------------
# Generate nanobot config
# ---------------------------------------------------------------------------
CONFIG_DIR="$HOME/.nanobot"
CONFIG_FILE="$CONFIG_DIR/config.json"
mkdir -p "$CONFIG_DIR"
mkdir -p "$WORKSPACE"

cat > "$CONFIG_FILE" << EOFCFG
{
  "agents": {
    "defaults": {
      "workspace": "$WORKSPACE",
      "model": "$MODEL",
      "provider": "custom",
      "maxTokens": 4096,
      "contextWindowTokens": 8192,
      "temperature": 0.7,
      "maxToolIterations": 50,
      "maxConcurrentSubagents": 4,
      "timezone": "Asia/Shanghai",
      "botName": "trainable-claw",
      "botIcon": "",
      "sessionTtlMinutes": 60,
      "disabledSkills": [
        "image-generation",
        "long-goal",
        "cron"
      ]
    }
  },
  "providers": {
    "custom": {
      "apiBase": "$API_BASE",
      "apiKey": "no-key"
    }
  },
  "gateway": {
    "host": "0.0.0.0",
    "port": $NANOBOT_PORT
  },
  "api": {
    "host": "127.0.0.1",
    "port": 8900
  },
  "tools": {
    "exec": {
      "sandbox": "none",
      "allowLocalhost": true
    }
  }
}
EOFCFG

echo "Config written to $CONFIG_FILE"
echo ""

# ---------------------------------------------------------------------------
# Check serve_ppo
# ---------------------------------------------------------------------------
echo "Checking serve_ppo at $API_BASE ..."
if curl -sf "${API_BASE}/health" > /dev/null 2>&1; then
    echo "  serve_ppo: OK"
else
    echo "  serve_ppo: NOT REACHABLE (continuing anyway)"
fi
echo ""

# ---------------------------------------------------------------------------
# Start nanobot gateway
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"
echo "Starting nanobot gateway on port $NANOBOT_PORT ..."
echo "  Log: /tmp/nanobot_gateway.log"

NANOBOT_PACKAGE="$PROJECT_DIR/nanobot-0.2.1"

nohup python -m nanobot gateway \
    --config "$CONFIG_FILE" \
    > /tmp/nanobot_gateway.log 2>&1 &

PID=$!
echo "  PID: $PID"
echo ""

# Wait for startup
sleep 3
if kill -0 "$PID" 2>/dev/null; then
    echo "nanobot gateway started successfully"
    echo ""
    echo "=== Quick Start ==="
    echo "  WebChat:  http://localhost:$NANOBOT_PORT/webui/"
    echo "  API:      http://localhost:$NANOBOT_PORT/api/v1/chat/completions"
    echo "  nanobot API: http://localhost:8900/v1/chat/completions"
    echo ""
    echo "  Monitor:  tail -f /tmp/nanobot_gateway.log"
else
    echo "ERROR: nanobot gateway failed to start"
    echo "Check log: tail -f /tmp/nanobot_gateway.log"
    exit 1
fi
