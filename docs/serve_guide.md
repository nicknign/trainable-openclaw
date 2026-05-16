# A1+A2 启动与调试指南

## 前提条件

- **环境**: Linux (Ubuntu 20.04+), CUDA 12.x, 8×GPU (最低 1×GPU 也可测试)
- **Python**: 3.10+, 已安装 veRL 依赖 (`pip install -e .` 在 verl-main-0516 目录)
- **模型**: 任意 HuggingFace 格式的 LLM（如 `Qwen/Qwen2.5-1.5B-Instruct`）

## 快速验证（无 GPU，Windows/Linux 通用）

在部署 Linux 之前，先在本地跑通全部冒烟测试：

```bash
cd projects/trainable-openclaw
pip install pytest fastapi uvicorn pydantic numpy
python -m pytest tests/ -v
# 预期: 33 passed (9 A1 + 24 A2)
```

---

## 第一步：启动推理服务

### 1.1 最小启动命令

```bash
cd projects/trainable-openclaw

python -m verl.trainer.serve_ppo \
    actor_rollout_ref.model.path=/path/to/your/model \
    actor_rollout_ref.rollout.name=vllm \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.serve_port=8000 \
    trainer.project_name=trainable_openclaw \
    trainer.experiment_name=serve_test \
    trainer.logger='[console]' \
    data.train_files= \
    data.val_files= \
    actor_rollout_ref.rollout.load_format=dummy \
    actor_rollout_ref.model.use_shm=true
```

### 1.2 参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `actor_rollout_ref.model.path` | 模型路径 (HF 格式) | **必填** |
| `actor_rollout_ref.rollout.name` | 推理引擎: vllm / sglang / hf | `vllm` |
| `trainer.n_gpus_per_node` | 每节点 GPU 数 | 实际数量 |
| `trainer.serve_port` | API 端口 | `8000` |
| `trainer.logger` | 日志输出 | `'[console]'` 即可 |
| `data.train_files` | 置空 (serve 模式不需要) | 空字符串 |
| `data.val_files` | 置空 | 空字符串 |
| `actor_rollout_ref.rollout.load_format` | 模型加载方式 | `dummy`=随机初始化 / `hf`=HF 权重 |
| `actor_rollout_ref.model.use_shm` | 使用共享内存加速加载 | `true` |

### 1.3 测试用小模型示例（1 GPU）

```bash
python -m verl.trainer.serve_ppo \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.rollout.name=vllm \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.serve_port=8000 \
    trainer.project_name=test \
    trainer.experiment_name=test \
    trainer.logger='[console]' \
    data.train_files= \
    data.val_files= \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.model.use_shm=false
```

### 1.4 启动成功标志

```
============================================================
  veRL Inference Server starting on http://0.0.0.0:8000
  Health check: http://localhost:8000/v1/health
  Chat API:     http://localhost:8000/v1/chat/completions
============================================================
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Training orchestrator started — idle_timeout=60.0s, min_samples=16
```

---

## 第二步：验证基础功能

### 2.1 Health Check

```bash
curl -s http://localhost:8000/v1/health | python -m json.tool
```

预期输出（正常推理中）：
```json
{
    "status": "ok",
    "mode": "serving",
    "uptime_seconds": 123.4,
    "active_requests": 0,
    "gpu_count": 8
}
```

关键字段：
- `mode: "serving"` — 当前处于推理模式（非训练）
- `active_requests: 0` — 当前无正在处理的请求
- `gpu_count: 8` — GPU 数量

### 2.2 发一个推理请求

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.7,
    "max_tokens": 100
  }' | python -m json.tool
```

预期输出：
```json
{
    "id": "chatcmpl-xxxxxxxxxxxx",
    "object": "chat.completion",
    "created": 1715000000,
    "model": "test",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "你好！有什么我可以帮助你的吗？"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 12,
        "total_tokens": 17
    }
}
```

### 2.3 验证多次请求有不同采样结果

```bash
# 发两次相同请求，输出应不同（采样非 greedy）
for i in 1 2; do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"用一句话介绍你自己"}],"temperature":1.0}' \
    | python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
done
```

### 2.4 验证 sampling 参数回退

```bash
# 不传 temperature，应使用 rollout_config 默认值
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}]}' \
  | python -m json.tool
```

---

## 第三步：验证 A2 空闲检测 + 训练触发

### 3.1 核心流程

```
请求 → chat_completions
 ├─ training_in_progress? → 503
 ├─ generate → 返回用户
 └─ orchestrator.record_request() 入队

后台监控(每1s):
 └─ should_train()?
     ├─ idle_timeout (默认60s) 且
     └─ min_samples (默认16) ≥ 阈值
         └─ _train_bridge():
              ├─ sleep_replicas()
              ├─ train_step() [当前stub]
              └─ wake_replicas()
```

### 3.2 测试：训练期间 503

用低阈值快速触发训练：

```bash
python -m verl.trainer.serve_ppo \
    actor_rollout_ref.model.path=... \
    trainer.serve_port=8000 \
    trainer.idle_timeout=5 \
    trainer.min_samples=2 \
    ...  # 其他参数同上
```

然后：
```bash
# 1. 快速发 2 个请求（满足 min_samples=2）
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"1+1=?"}]}' > /dev/null
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"2+2=?"}]}' > /dev/null

# 2. 等 5 秒后（idle_timeout=5），训练被触发
sleep 6

# 3. 训练期间发请求 → 预期 503
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
# 预期: {"detail": "Training in progress, try later"}

# 4. 查看日志应包含:
#    "Training triggered — 2 samples, idle=5.xs"
#    "train_step called with 2 samples (stub)"
#    "Training complete in x.xs, resuming inference"
```

### 3.3 观察日志关键字

```bash
# 启动时
"Training orchestrator started — idle_timeout=5.0s, min_samples=2"

# 请求到达（不显式打印，通过 active_requests 间接观察）
# GET /v1/health → active_requests 字段

# 训练触发
"Training triggered — 2 samples, idle=5.xs"
"All rollout replicas asleep"
"train_step called with 2 samples (stub)"
"All rollout replicas awake"
"Training complete in x.xs, resuming inference"

# 训练失败（错误处理）
"Training failed — resuming inference with old weights"
```

---

## 第四步：调试检查清单

### 4.1 服务没起来

| 症状 | 可能原因 | 解决 |
|------|---------|------|
| `ModuleNotFoundError: No module named 'verl'` | PYTHONPATH 没设 | `export PYTHONPATH=verl-main-0516:$PYTHONPATH` |
| `ModuleNotFoundError: No module named 'trainable_openclaw'` | 包路径没设 | `export PYTHONPATH=.:$PYTHONPATH` |
| `CUDA out of memory` | GPU 显存不足 | 减小 `gpu_memory_utilization`（如 0.4），或减小 TP 数 |
| `ray init` 失败 | Ray 未安装/版本不匹配 | `pip install ray` 或检查版本 |
| vLLM 启动超时 | 模型太大/网络慢 | 先用小模型测试，设置 `ray_wait_register_center_timeout=600` |

### 4.2 请求失败

| 症状 | 可能原因 | 解决 |
|------|---------|------|
| 503 `Server not initialized yet` | LLM 还在加载 | 等一会再试，看 `health` 的 `uptime_seconds` |
| 503 `Training in progress` | 正在训练 | 正常行为，训练完自动恢复 |
| 500 内部错误 | 生成异常 | `grep "Generation failed"` 查看具体错误 |
| 超时无响应 | 队列堆积 | `/v1/health` 查看 `active_requests` |

### 4.3 训练没触发

| 症状 | 可能原因 | 解决 |
|------|---------|------|
| 等了很久没触发 | idle_timeout 未满足 | 确认无人发请求，`/v1/health` 看 `uptime` |
| 样本够了但不触发 | min_samples 不够 | `grep "should_train"` 或检查 `orchestrator.sample_count` |
| 触发但报错 | train_step stub 报错 | 查看错误信息，当前 stub 不会报错 |

### 4.4 查看运行时状态

```bash
# 连续监控
watch -n 1 'curl -s http://localhost:8000/v1/health | python -m json.tool'

# 或者写一个简单的监控脚本
while true; do
  echo "$(date -Iseconds)  $(curl -s http://localhost:8000/v1/health)"
  sleep 2
done
```

---

## 第五步：完整端到端验证脚本

创建 `scripts/verify_serve.sh`：

```bash
#!/bin/bash
set -euo pipefail

HOST="http://localhost:8000"
PASS=0
FAIL=0

check() {
    local desc="$1"
    local expected="$2"
    local actual="$3"
    if echo "$actual" | grep -q "$expected"; then
        echo "  PASS: $desc"
        ((PASS++))
    else
        echo "  FAIL: $desc (expected '$expected', got '$actual')"
        ((FAIL++))
    fi
}

echo "=== A1: Health Check ==="
HEALTH=$(curl -s $HOST/v1/health)
echo "$HEALTH" | python -m json.tool
check "status=ok"     '"status":"ok"' "$HEALTH"
check "mode=serving"  '"mode":"serving"' "$HEALTH"

echo ""
echo "=== A1: Chat Completions ==="
CHAT=$(curl -s $HOST/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"temperature":0.7,"max_tokens":50}')
echo "$CHAT" | python -m json.tool
check "object field"        '"object":"chat.completion"' "$CHAT"
check "choices present"     '"choices"' "$CHAT"
check "usage present"       '"usage"' "$CHAT"
check "finish_reason"       '"finish_reason":"stop"' "$CHAT"

echo ""
echo "=== A1: Uninitialized 503 (test-only, skip in production) ==="

echo ""
echo "=== A2: Training Mode 503 ==="
# 在低阈值配置下（idle_timeout=5, min_samples=2），
# 快速发 2 个请求后等待 6 秒触发训练，然后发请求验证 503
# （此步骤需要在服务端配置低阈值，此处仅演示）
echo "  (需要服务端配置 idle_timeout=5 min_samples=2)"
echo "  curl 应返回: {\"detail\": \"Training in progress, try later\"}"

echo ""
echo "=== Summary: $PASS passed, $FAIL failed ==="
```

使用方法：
```bash
chmod +x scripts/verify_serve.sh
./scripts/verify_serve.sh
```

---

## 文件速查

| 文件 | 角色 |
|------|------|
| `verl-main-0516/verl/trainer/serve_ppo.py` | 生产入口：Ray + ServeRunner + 启动逻辑 |
| `trainable_openclaw/server/api.py` | API 层：Pydantic 模型 + FastAPI endpoint（可脱离 GPU 测试） |
| `trainable_openclaw/training/orchestrator.py` | A2：空闲检测 + 样本队列 + 训练触发 |
| `tests/test_serve_ppo.py` | A1 冒烟测试（9 个，Windows 可跑） |
| `tests/test_orchestrator.py` | A2 冒烟测试（24 个，Windows 可跑） |

## 当前已知限制 (A2 stub)

- `train_step()` 是 stub，只打印日志，不执行实际训练
- 实际训练需要接入 `RayPPOTrainer.fit()` 单步逻辑 → A3 完成
- 训练期间请求直接 503，无排队机制 → 后续优化
