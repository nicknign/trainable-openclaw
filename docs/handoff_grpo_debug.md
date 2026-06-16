# GRPO Training Handoff — 2026-06-16

## 目录

1. [目标与背景](#1-目标与背景)
2. [环境准备（新机器首次设置）](#2-环境准备新机器首次设置)
3. [推理复现：跑 baseline 验结果](#3-推理复现跑-baseline-验结果)
4. [代码改动清单](#4-代码改动清单)
5. [数据说明](#5-数据说明)
6. [训练流程](#6-训练流程)
7. [4B 训练：内存分析与可行方案](#7-4b-训练内存分析与可行方案)
8. [多机训练](#8-多机训练)
9. [遇到的错误和解决方案](#9-遇到的错误和解决方案)
10. [调试优先级](#10-调试优先级)

---

## 1. 目标与背景

训练 Qwen3.5 模型在 tau-bench retail 客服任务上通过 GRPO + Agent Loop 学会多轮工具调用。

**当前 baseline**:
| 配置 | Completion | 说明 |
|------|-----------|------|
| Qwen3.5-2B | 0% | 无法正确输出工具调用 |
| Qwen3.5-4B + hermes parser | 6.67% (2/30) | 旧 baseline |
| **Qwen3.5-4B + qwen3_coder** | **45.95% (17/37)** | 当前 SOTA baseline |
| DeepSeek V4-Flash (上限) | 35.2% (2/10) | 零基础设施错误 |

**核心矛盾**: 2B 完全不输出合法 tool calls → GRPO rollout 只有 EOS → 无法训练。必须先让 2B 学会基本格式（SFT预热），或用 4B 训练。

---

## 2. 环境准备（新机器首次设置）

### 2.1 目录结构

```
/data/
├── models/
│   ├── Qwen3.5-2B/       # 2B 推理用
│   └── Qwen3.5-4B/       # 4B 训练用
├── wangye/
│   └── trainable-openclaw/
│       ├── verl-main-0516/           # verl 源码（不装包，直接改）
│       ├── trainable_openclaw/       # 项目 Python 库
│       ├── scripts/
│       ├── ai_scripts/
│       ├── data/
│       └── nanobot-0.2.1/           # nanobot 源码
└── anaconda3/            # Python 3.13, CUDA 12.8
```

### 2.2 必需环境变量

```bash
# 每个新 shell 都要设置（可写到 ~/.bashrc）
export LD_LIBRARY_PATH="/data/anaconda3/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/data/wangye/trainable-openclaw:/data/wangye/trainable-openclaw/verl-main-0516:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export DEEPSEEK_API_KEY="你的 DeepSeek API key"  # baseline eval 需要
```

**说明**:
- `LD_LIBRARY_PATH`: libstdc++ 版本兼容
- `PYTHONPATH`: 项目模块 + verl 源码（不装包，直接 import 源码）
- `verl` 的代码在 `verl-main-0516/verl/` 下，修改后立即生效，无需 pip install

### 2.3 安装依赖

```bash
/data/anaconda3/bin/pip install paramiko aiohttp dulwich
```

### 2.4 验证环境

```bash
cd /data/wangye/trainable-openclaw
/data/anaconda3/bin/python ai_scripts/remote_diag.py
```

预期输出: 6 项全部 `[OK]`（数据文件、agent_tools、grpo_reward、verl 版本、模型、CUDA）。

---

## 3. 推理复现：跑 baseline 验结果

**目标**: 用 nanobot + vllm 跑一遍 eval，确认 2B 的 0% 和 4B 的 45.95% 可以复现。

### 3.1 架构

```
vllm :8000 (Qwen3.5-4B/2B, OpenAI API)
  ← nanobot API :8900 (管理多轮对话 + 工具调用)
    ← NanobotEvaluator (发送用户消息, 接收 agent 回复)
      ↔ SimulatedUser (DeepSeek-chat, 模拟客户)
```

### 3.2 Step 1: 启动 vllm (serve_ppo)

```bash
cd /data/wangye/trainable-openclaw

# 4B 模型（推荐先复现这个）
MODEL_PATH=/data/models/Qwen3.5-4B MAX_MODEL_LEN=49152 GPU_MEM=0.85 \
  bash scripts/deploy/start_experience.sh

# 验证 vllm 就绪
curl -s http://localhost:8000/v1/health
# 预期: {"status":"ok","mode":"serving",...}

# 验证推理可用
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":50}'
# 预期: choices[0].message.content 有正常回复
```

**start_experience.sh 做的事**:
1. 杀掉旧 vllm/nanobot 进程
2. 用 `python -m verl.trainer.serve_ppo` 启动推理服务（:8000）
3. 生成 nanobot config（`/root/.nanobot/config.json`）
4. 启动 nanobot API（:8900）+ Gateway（:18790）

**关键参数**:
- `MAX_MODEL_LEN=49152`: 4B 用 48K context；2B 最大 32K
- `GPU_MEM=0.85`: 单服务推理，可以大到 0.85；训练时混合引擎只能用 0.4-0.5

### 3.3 Step 2: 测试 nanobot API

```bash
# 健康检查
curl -s http://localhost:8900/health

# 发一条简单消息（无工具）
curl -s http://localhost:8900/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

### 3.4 Step 3: 跑单任务 eval

```bash
cd /data/wangye/trainable-openclaw

# 跑第 1 个测试任务
PYTHONPATH="nanobot-0.2.1:$PYTHONPATH" /data/anaconda3/bin/python \
  ai_scripts/run_nanobot_eval.py --task 0

# 跑指定 task ID
PYTHONPATH="nanobot-0.2.1:$PYTHONPATH" /data/anaconda3/bin/python \
  ai_scripts/run_nanobot_eval.py --task-id retail_task_40
```

**观察重点**:
- agent 回复中是否有 tool call 格式（`<tool_call>` 标签）
- nanobot 是否正确解析并执行工具
- 对话轮次是否正常推进

### 3.5 Step 4: 跑全量 retail eval（37 条）

```bash
# 需要先有批量运行脚本 —— 目前只有单任务脚本
# 简易批量方式：
for i in $(seq 0 36); do
  echo "=== Task $i ==="
  PYTHONPATH="nanobot-0.2.1:$PYTHONPATH" /data/anaconda3/bin/python \
    ai_scripts/run_nanobot_eval.py --task $i --max-rounds 10
done
```

**日志路径**: `/tmp/serve_ppo_experience.log`, `/tmp/nanobot_api.log`

### 3.6 Step 5: 切换模型验证 2B

```bash
# 先停掉 4B
pkill -f serve_ppo; pkill -f nanobot; sleep 3

# 用 2B 重启
MODEL_PATH=/data/models/Qwen3.5-2B MAX_MODEL_LEN=32768 GPU_MEM=0.85 \
  bash scripts/deploy/start_experience.sh

# 再跑 eval
PYTHONPATH="nanobot-0.2.1:$PYTHONPATH" /data/anaconda3/bin/python \
  ai_scripts/run_nanobot_eval.py --task 0
```

**预期**: 2B 的 nanobot eval 应该 0/37 或接近 0%——agent 要么不输出 tool call，要么输出的格式 nanobot 解析失败。

### 3.7 停止服务

```bash
pkill -f serve_ppo; pkill -f nanobot
```

---

## 4. 代码改动清单

以下改动已经应用到本地的 `verl-main-0516/` 和项目目录中。

### 改动 1: bucketed_weight_transfer.py — PyTorch 2.10 IPC 兼容

**文件**: `verl-main-0516/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`  
**行号**: 48-51

**为什么需要**: PyTorch 2.10 改变了 IPC shared memory handle 的序列化格式。原来 `list_args` 固定有 7 个元素，新版本减少到 6 个。当 `device_id` 为 `None` 时（FSDP 模式下），代码仍然尝试访问 `list_args[6]`，造成 `IndexError`。

**改动前**:
```python
if device_id is not None:
    list_args[6] = device_id
```

**改动后**:
```python
if device_id is not None and len(list_args) > 6:
    list_args[6] = device_id
```

**影响范围**: 仅在用 `bucketed_weight_transfer` 传输 LoRA 权重时触发（hybrid engine 的 vLLM ↔ FSDP 权重同步）。

---

### 改动 2: add_agent_name.py — 添加必需的 data 字段

**文件**: `scripts/data/add_agent_name.py`  
**行号**: 58-60

**为什么需要**: verl 的 naive reward manager 要求每个数据记录必须包含:
- `data_source`: 用于数据分组（wande 日志用）
- `reward_model.ground_truth`: 用于选择 reward 计算逻辑

缺少任一字段都会 `KeyError`。

**改动**: 生成时自动从原始字段推导:
```python
obj["data_source"] = obj.get("source", "taubench_retail")
obj["reward_model"] = {"ground_truth": obj.get("domain", "retail"), "style": "rule"}
```

**使用**:
```bash
python scripts/data/add_agent_name.py data/tau_bench/train_split_66.jsonl data/tau_bench/train_agent_66.jsonl
python scripts/data/add_agent_name.py data/tau_bench/val_split_18.jsonl data/tau_bench/val_agent_18.jsonl
```

---

### 改动 3: grpo_retail.yaml — 训练配置

**文件**: `scripts/train/grpo_retail.yaml` + `verl-main-0516/verl/trainer/config/grpo_retail.yaml`  
（两个文件内容相同，前者是项目的，后者是 Hydra config_path 需要的）

**核心参数和含义**:

```yaml
data:
  # prompt + system prompt + tool defs ≈ 2700 tokens，用 4096 给安全余量
  max_prompt_length: 4096
  # 多轮对话（2 turns user + 2 turns assistant）× tool responses，8192 够用
  max_response_length: 8192
  # 如果 prompt 超长，截断左边（保留末尾的任务说明）
  truncation: left
  train_batch_size: 2

actor_rollout_ref:
  model:
    path: /data/models/Qwen3.5-2B  # 改成 /data/models/Qwen3.5-4B 用 4B
    lora_rank: 64
    lora_alpha: 128
    # 只选未 fused 的层。Qwen3 的 in_proj/up_proj/gate_proj 在 vLLM 注册时已
    # fuse 成 in_proj_qkvz / gate_up_proj，LoRA target 这些会 shape mismatch
    target_modules:
      - out_proj
      - down_proj

  rollout:
    name: vllm
    mode: async          # Agent Loop 必须用 async
    n: 4                 # 每个 prompt 生成 4 条候选
    temperature: 0.9
    # ⚠️ 以下两个值必须和 data.xxx 一致
    prompt_length: 4096
    response_length: 8192
    gpu_memory_utilization: 0.50   # 2B: 0.50 够用；4B: 0.35-0.40
    enforce_eager: true  # 禁用 CUDA graphs，避免 Qwen3 LoRA IndexError
    calculate_log_probs: true
    multi_turn:
      enable: True
      max_user_turns: 2       # 最多 2 轮用户交互
      max_assistant_turns: 2  # 最多 2 轮 agent 回复
      format: qwen3_coder     # Qwen3 原生 XML tool call
      function_tool_path: /data/wangye/trainable-openclaw/trainable_openclaw/training/agent_tools.py
      max_tool_response_length: 1024
    agent:
      num_workers: 2
      default_agent_loop: tool_agent

  actor:
    ppo_mini_batch_size: 2
    ppo_epochs: 1
    optim:
      lr: 2.0e-5
    kl_loss_coef: 0.0   # 关闭 reference policy 以省显存

algorithm:
  adv_estimator: grpo   # 不用 critic，不需要 value 估算

trainer:
  total_training_steps: 200
  save_freq: 50
  test_freq: 50
  n_gpus_per_node: 1    # 单卡
  nnodes: 1             # 单机
```

**⚠️ 关键约束**: `data.max_prompt_length == rollout.prompt_length` 且 `data.max_response_length == rollout.response_length`，否则 tensor 维度不匹配。

---

### 改动 4: agent_loop.py truncation patches（未应用到本地）

**为什么没有应用**: 远程 verl（v0.9.0.dev latest commit）和本地 verl-main-0516 版本不同。远程有 `_pad_token_ids()` 辅助方法，本地没有。

远程应用的 patch 位置在 `AgentLoopBase._agent_loop_postprocess()` 方法中，功能是在 tokenizer padding 之前做一次 truncation（因为 `tokenizer.pad()` 只做 padding 不做 truncation）：

```python
# 安全上限：防止模型生成超过 response_length 的 token
response_ids = output.response_ids[:self.rollout_config.response_length]
response_mask_data = output.response_mask[:self.rollout_config.response_length]
```

如果你在本地遇到类似的 "response 长度超过 max_length 导致 tensor 不一致" 问题，需要在 `_agent_loop_postprocess` 中添加相同的 truncation，但改用 `self.tokenizer.pad()`。

---

## 5. 数据说明

### 5.1 数据来源

tau-bench retail — 客服对话任务。用户（模拟）扮演客户，agent 扮演客服。场景包括订单查询、退货、换货、退款、地址修改、促销匹配等。

### 5.2 数据流转

```
tau-bench 原始定义 (164 retail + 164 airline tasks)
    │
    ├─→ scripts/data/convert_tau_bench.py      → 原始 JSONL
    ├─→ scripts/data/filter_split_tau_bench.py  → train/val/test 按 task_id 分
    ├─→ scripts/data/augment_data.py           → 每 task 3-4 个 variant
    └─→ scripts/data/add_agent_name.py          → Agent Loop 格式
    │
    ▼
train_agent_66.jsonl  ─── 251 条, 66 unique tasks  ─── GRPO 训练
val_agent_18.jsonl    ───  69 条, 18 unique tasks  ─── 训练中验证
test_agent_37.jsonl   ───  37 条, 23 unique tasks  ─── baseline eval (不使用)
```

### 5.3 数据隔离

**按 task_id 拆分**: 同一个 task 的所有 variant 只在一个 split 中，零泄漏。

train: `retail_task_*` 中随机 66 个  
val: 随机 18 个（不与 train 重叠）  
test: 随机 37 个（23 unique tasks，不与 train/val 重叠）

### 5.4 每条数据格式

```json
{
    "id": "retail_task_12_v0",
    "source": "taubench_retail",
    "data_source": "taubench_retail",
    "task_id": "retail_task_12",
    "domain": "retail",
    "variant": "v0",
    "evaluation": {
        "rubric": "Check that agent looked up order...",
        "expected_outcome": "Order cancelled successfully"
    },
    "tools": ["lookup_user", "lookup_order", "cancel_order"],
    "prompt": [
        {
            "role": "system",
            "content": "You are a retail customer service agent..."
        },
        {
            "role": "user",
            "content": "Hi, I need help with my order..."
        }
    ],
    "agent_name": "tool_agent",
    "reward_model": {"ground_truth": "retail", "style": "rule"}
}
```

**关键字段说明**:

| 字段 | 用途 | 谁使用 |
|------|------|--------|
| `agent_name` | "tool_agent" → verl 用 ToolAgentLoop 处理多轮工具调用 | verl Agent Loop |
| `data_source` | 数据分组标识 | naive reward manager |
| `reward_model.ground_truth` | "retail" → 选择零售专属 rubric engine | reward function |
| `prompt` | messages 列表（非纯文本） | verl 内部调用 `apply_chat_template` + tools |
| `tools` | 允许使用的工具列表 | Agent Loop 注入 tool definitions |

### 5.5 工具系统

34 个工具注册在 `trainable_openclaw/training/agent_tools.py`，用 `@function_tool` 装饰器：

```python
# 例子：查询用户信息
@function_tool
def lookup_user(user_id: str) -> dict:
    """Look up a customer by their user ID."""
    ...

# 例子：查询订单
@function_tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by its ID."""
    ...
```

verl Agent Loop 在 rollout 时加载这些工具定义，按 `qwen3_coder` XML 格式注入 prompt：

```
<tool_call>
{"name": "lookup_user", "arguments": {"user_id": "U12345"}}
</tool_call>
```

---

## 6. 训练流程

### 6.1 启动命令

```bash
export LD_LIBRARY_PATH="/data/anaconda3/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/data/wangye/trainable-openclaw:/data/wangye/trainable-openclaw/verl-main-0516:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export REWARD_MODE=agent
export RAY_DISABLE_DOCKER_CPU_WARNING=1

cd /data/wangye/trainable-openclaw/verl-main-0516

nohup /data/anaconda3/bin/python -m verl.trainer.main_ppo \
    --config-name grpo_retail \
    > /tmp/grpo_agent.log 2>&1 &
```

或直接用 `bash ai_scripts/launch_training.sh`。

### 6.2 训练流程

```
Step 1: Ray 初始化 + vLLM 启动
  ├─ Ray cluster（单机 local mode）
  ├─ WorkerDict (FSDP actor trainer)
  └─ vLLMHttpServer (rollout engine)

Step 2: 第一轮 rollout (每 step 重复)
  ├─ 加载 2 prompts × 4 rollouts = 8 个并发请求
  ├─ Agent Loop: model generate → parse tool call → execute tool →
  │   append observation → model generate → ... → complete
  └─ 收集 trajectories + rewards

Step 3: Actor 训练
  ├─ vLLM sleep (释放 KV cache 给 actor)
  ├─ compute GRPO advantage (基于 reward)
  ├─ LoRA forward/backward (FSDP)
  └─ vLLM wake (恢复 KV cache)

Step 4: 重复 Step 2-3，共 200 步
  ├─ 每 50 步 save checkpoint
  └─ 每 50 步 validation
```

### 6.3 监控命令

```bash
# 进程是否存活
ps aux | grep main_ppo | grep -v grep

# GPU 占用
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# 日志实时
tail -f /tmp/grpo_agent.log

# 只看关键事件（过滤掉 vLLM 的视觉层警告）
tail -f /tmp/grpo_agent.log | grep -v 'qwen3_5.py\|visual.blocks\|will be ignored\|linear_attn\|layer_type'

# 看 reward 值
grep -i 'reward\|score' /tmp/grpo_agent.log
```

### 6.4 训练成功的标志

1. `Training Progress: 1/200` 出现（第一步完成）
2. reward 值在 0-1 之间变化（GRPO 学习信号）
3. 每步 rollout + training 时间稳定（2B ≈ 2-3 min/step, 4B ≈ 5-8 min/step）
4. 无 CUDA OOM 或 split 错误

---

## 7. 4B 训练：内存分析与可行方案

### 7.1 为什么 4B 在 48GB 上 OOM

Qwen3.5-4B 的 hybrid engine 内存分布（RTX 4090 48GB）：

```
vLLM engine (rollout 时活跃):
  ├─ 模型权重 (bf16):               ~8 GB
  ├─ LoRA adapter 权重:             ~0.1 GB
  ├─ KV cache (4096+8192=12288 tokens, n=4, gpu_mem_util=0.35):
  │   ~4.2 GB (12288 × 32 layers × 2 KV × 128 heads × 128 dim × 2 bytes × 4 seqs)
  ├─ vLLM overhead (CUDA graphs, etc): ~2 GB
  └─ 小计:                          ~14 GB

actor (训练时活跃):
  ├─ 模型权重 (bf16):               ~8 GB
  ├─ LoRA adapter 权重:             ~0.1 GB
  ├─ 梯度:                          ~0.1 GB (仅 LoRA 参数)
  ├─ 优化器状态 (AdamW):            ~0.2 GB (仅 LoRA 参数)
  ├─ 激活 (FSDP + gradient checkpoint): ~8-12 GB (取决于 batch size)
  └─ 小计:                          ~17-21 GB

hybrid engine wake_up (vLLM 重新分配 KV cache 时):
  ├─ 模型权重 + LoRA 仍在 GPU
  ├─ actor 的 optimizer state 和激活在 CPU（offload 后）
  ├─ vLLM cumem_allocator 分配新的 KV cache blocks
  └─ 峰值时刻: 模型权重(~8GB) + LoRA(~0.1GB) + 新KV block池(~4GB)
                + vLLM overhead(~2GB) + 残留actor state(~1-2GB)
               ≈ 15-17 GB  ← 理论上够用

但实际 OOM 发生在 cumem_allocator 重新分配时:
  - vLLM 的 CUDACachingAllocator 和 PyTorch 的 CUDA 内存管理冲突
  - 碎片化导致无法分配连续的 KV cache blocks
  - sleep_level=1 只释放 KV cache，不释放模型权重 → 碎片恶化
```

### 7.2 试过的方案

| 方案 | gpu_mem | 结果 |
|------|---------|------|
| 4B + LoRA rank=16 + prompt=1024 + response=12288 | 0.50 | CUDA OOM during wake_up |
| 4B + LoRA rank=16 + 同上 | 0.35 | CUDA OOM during wake_up |
| 4B + LoRA rank=16 + 同上 | 0.25 | "No available memory for cache blocks" — block 池太小 |
| 2B + LoRA rank=64 + prompt=4096 + response=8192 | 0.50 | 不 OOM，但 `split_sizes=[4,4,...]` |

### 7.3 让 4B 能跑的可行方向

**方案 A: 减小 context 窗口（最简单）**
```yaml
data:
  max_prompt_length: 2048    # 从 4096 减到 2048
  max_response_length: 4096  # 从 8192 减到 4096
rollout:
  prompt_length: 2048
  response_length: 4096
  gpu_memory_utilization: 0.40
```
代价: prompt 可能被截断，多轮对话空间变小。

**方案 B: 用 sleep_level=2**
在 agent_loop 配置中设为 level 2（释放更多显存），但可能会更慢（每次 wake 需要重新加载一些状态）。

**方案 C: CPU offload actor optimizer（verl 已支持）**
```yaml
actor:
  fsdp_config:
    optimizer_offload: true   # 把优化器状态放到 CPU
    param_offload: true       # 训练时把模型权重也放到 CPU
```
CPU 显存换 GPU 显存，但训练会慢 2-3×。

**方案 D: 多卡/多机**
见下节。

### 7.4 切换到 4B 需要改的地方

```bash
# 确保模型存在
ls /data/models/Qwen3.5-4B/config.json

# 改 config
# data.max_prompt_length: 2048-4096（根据显存情况）
# rollout.gpu_memory_utilization: 0.35-0.40
# 其他不变

# 训练启动命令不变
```

---

## 8. 多机训练

### 8.1 verl 的多节点支持

verl 通过 Ray cluster 支持多节点。关键配置：

```yaml
trainer:
  n_gpus_per_node: 8   # 每节点 GPU 数
  nnodes: 2            # 节点数
```

### 8.2 多节点设置步骤

**1) 启动 Ray head（主节点）**:
```bash
ray start --head --port=6379
```

**2) 启动 Ray worker（从节点）**:
```bash
ray start --address='<head_ip>:6379'
```

**3) 运行训练（在 head 节点）**:
```bash
# 需要确保所有节点都能访问相同的：
# - 模型路径（NFS 或每节点相同路径）
# - 数据路径（NFS 或每节点相同路径）
# - PYTHONPATH（verl 代码版本一致）

cd /data/wangye/trainable-openclaw/verl-main-0516
python -m verl.trainer.main_ppo --config-name grpo_retail
```

### 8.3 当前配置是单节点

```yaml
trainer:
  n_gpus_per_node: 1
  nnodes: 1
```

改成多机时改为:
```yaml
trainer:
  n_gpus_per_node: 1    # 每节点 1 张 4090
  nnodes: 2             # 2 台机器
```

### 8.4 多机时的注意事项

- **代码一致性**: 所有节点的 verl-main-0516 必须版本完全一致。用 rsync/git 同步。
- **模型路径**: 所有节点需要在相同路径有模型文件，或用 NFS。
- **数据路径**: 同上。如果用本地路径，每个节点都要有数据。
- **网络**: Ray 需要节点间通信端口（默认 6379，以及随机分配的其他端口）。
- **Agent Loop workers**: 会在各节点上分布启动，num_workers=2 意味着总共 2 个 worker（不是每节点 2 个）。

### 8.5 多机调试建议

先从单机 2B 跑通，再扩展:

1. **Phase 1**: 单机 2B，验证训练流程正确（reward 提升、无 crash）
2. **Phase 2**: 单机 4B + CPU offload，验证 4B 能训
3. **Phase 3**: 双机 4B，验证多机通信正常
4. **Phase 4**: 双机 4B × 2 GPU，扩大 batch 加速

---

## 9. 遇到的错误和解决方案

### Error 1: IndexError in rebuild_ipc
```
File ".../bucketed_weight_transfer.py", line 51
    list_args[6] = device_id
IndexError: list assignment index out of range
```
**根因**: PyTorch 2.10 IPC handle 格式从 7 元素变 6 元素。  
**修复**: 改动 1（已应用）。  
**状态**: ✅ 已解决。

### Error 2: KeyError: 'data_source'
**根因**: naive reward manager 内部读取 `data_source` 字段。  
**修复**: 改动 2，`add_agent_name.py` 添加 `data_source` 字段。  
**状态**: ✅ 已解决。

### Error 3: KeyError: 'reward_model'
**根因**: naive reward manager 需要 `reward_model.ground_truth`。  
**修复**: 改动 2，添加 `reward_model` 字段。  
**状态**: ✅ 已解决。

### Error 4: CUDA OOM (4B)
```
torch.OutOfMemoryError: CUDA out of memory during vLLM wake_up
```
**根因**: 4B + LoRA + KV cache 峰值超 48GB。  
**缓解**: 切换 2B → 改动 3 调整 `gpu_memory_utilization`。  
**状态**: ⬜ 4B 训练尚未跑通（见第 7 节可行方向）。

### Error 5: RuntimeError — split_sizes（未解决）
```
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 8192,
got split_sizes=[4,4,4,4,4,4,4,4]
```

**调用链**:
```
prepare_micro_batches
  → rearrange_micro_batches
    → index_select_tensor_dict
      → tensor.unbind()
        → torch.split(values, lengths_scalars)
```

**含义**: 8 条序列（batch_size=2 × n=4），每条只有 4 个有效 response token。`torch.split` 尝试按 response_mask 拆解 response tokens，但总 token 数 = 8×4 = 32，远小于 response_length=8192。

**根因分析**: 2B 模型生成极短的输出（几乎是纯 EOS），response_mask 中几乎全是 0（padding/tool-response），只有约 4 个 token 被标记为 LLM 生成。

**为什么 2B 不生成 tool calls**:
- Baseline eval 已确认 2B = 0% completion rate
- 即使 prompt 完整（4096 tokens），模型也无法理解 qwen3_coder tool call 格式
- 输出 `<tool_call>` 但格式/参数错误，或直接输出 EOS

**可能的修复方向**:
1. **SFT 预热 2B**: 用 4B 在 nanobot 上的 successful trajectories 做 SFT dataset，让 2B 先学会基本格式
2. **换 4B**: 用上一节的内存优化方案
3. **简化工具格式**: 用 prompt engineering 而非复杂的 XML tool call

### Error 6: 远程和本地 verl 版本不同

远程机器克隆的是 verl main 分支最新版，本地 `verl-main-0516/` 是 5月16日的快照。差异：
- 远程: 有 `_post_init_prompt` 内置 prompt 截断
- 本地: 没有

**解决方案**: 保持一致。建议在公司机器上也 git clone 最新版，然后把改动 1 和改动 3 应用到新版本。

---

## 10. 调试优先级

按这个顺序调试，每一步验证通过再进入下一步：

### Priority 1: 复现 baseline
```
1. 启动 serve_ppo (vllm) → curl /v1/health 确认就绪
2. 启动 nanobot → curl /health 确认就绪
3. 跑 1 个 retail 任务 eval → 确认 agent-user 对话正常
4. 切换 4B → 跑 37 个 retail 任务 → 确认 ~46% completion
5. 切换 2B → 跑同样任务 → 确认 ~0% completion
```
**验证标准**: curl 能正常返回，nanobot eval 能看到对话日志。

### Priority 2: 2B 单步 rollout 验证
```
1. 单独用 vllm 加载 2B（不经过 nanobot）
2. 发一条带完整 tool definitions 的 prompt
3. 看模型原始输出（token IDs 和 decode 后的文本）
4. 确认是否输出 EOS 还是尝试输出 tool call
```
**验证标准**: 能看到模型的原始输出，判断是格式问题还是完全不理解。

### Priority 3: SFT 预热（如需要）
如果 Priority 2 确认 2B 完全不输出 tool call:
```
1. 收集 4B 的 successful trajectories（从 baseline 日志中提取）
2. 构造 SFT dataset（prompt → tool call + response）
3. QLoRA SFT 2B 2-3 epochs
4. 重新验证 2B 能否输出合法 tool call
5. 如果可以 → 开始 GRPO 训练
```

### Priority 4: 4B 直训（跳过 SFT）
如果 4B 已经有 46% completion，不需要 SFT:
```
1. 改 grpo_retail.yaml → model path 指向 4B
2. 减小 context (prompt=2048, response=4096)
3. 降低 gpu_memory_utilization (0.35)
4. 开启 CPU offload
5. 启动训练
```

### Priority 5: 多机扩展
```
1. 单机 2B 验证 GRPO 流程完整（reward 提升）
2. 双机设置 Ray cluster
3. 双机 4B 训练
```

---

## 附录：相关文件速查

| 文件 | 说明 |
|------|------|
| `scripts/train/grpo_retail.yaml` | 训练配置 (主) |
| `verl-main-0516/verl/trainer/config/grpo_retail.yaml` | Hydra 用的副本 |
| `scripts/deploy/start_experience.sh` | 启动 vllm + nanobot |
| `ai_scripts/run_nanobot_eval.py` | nanobot 单任务 eval |
| `ai_scripts/launch_training.sh` | 训练启动脚本 |
| `ai_scripts/remote_diag.py` | 环境诊断 |
| `trainable_openclaw/training/agent_tools.py` | 34 个 @function_tool |
| `trainable_openclaw/training/grpo_reward.py` | GRPO reward 函数 |
| `scripts/data/add_agent_name.py` | 数据预处理 |
| `scripts/tools/autodl_sync.py` | 远程同步工具 |
| `data/tau_bench/train_agent_66.jsonl` | 训练数据 (251 条) |
| `data/tau_bench/val_agent_18.jsonl` | 验证数据 (69 条) |
| `data/tau_bench/test_prompts_augmented.jsonl` | 测试数据 (baseline 用) |
| `docs/serve_guide.md` | vllm 推理服务部署指南 |
| `docs/plan.md` | 完整实验计划 |
| `verl-main-0516/verl/experimental/agent_loop/agent_loop.py` | Agent Loop 核心逻辑 |

## 附录：同步到远程

```bash
cd C:/work/code/claude-code/projects/trainable-openclaw

# 全量同步代码到远程
python scripts/tools/autodl_sync.py --full

# 远程执行命令
python scripts/tools/autodl_sync.py --exec "nvidia-smi"

# 查看远程日志
python scripts/tools/autodl_sync.py --tail /tmp/grpo_agent.log
```

SFTP 配置在 `.vscode/sftp.json`。
