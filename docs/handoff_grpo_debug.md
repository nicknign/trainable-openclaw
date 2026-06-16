# GRPO Training Handoff — 2026-06-16

## 目标

训练 Qwen3.5-2B 在 tau-bench retail 客服任务上通过 GRPO + Agent Loop 学会多轮工具调用。

## 环境

- **机器**: AutoDL, RTX 4090 48GB, CUDA 12.8
- **Python**: /data/anaconda3/bin/python (3.13)
- **PyTorch**: 2.10.0+cu128
- **verl**: v0.9.0.dev (从 GitHub clone 的 `verl-main-0516/`)
- **模型**: Qwen3.5-2B (路径 `/data/models/Qwen3.5-2B`)
- **连接**: `ssh -p 35865 root@connect.westc.seetacloud.com`, 密码 `ZZ4U+p9b3kL7`

## 启动命令

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

或者用 `ai_scripts/launch_training.sh` 一键启动。

## 代码改动

### 1. verl-main-0516/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py (行 48-51)

**问题**: PyTorch 2.10 改变了 IPC handle 格式，`list_args[6] = device_id` 在 device_id 为 None 时仍然访问不存在的索引。

**修复**: 加 `len(list_args) > 6` 保护。

```python
# 修改前
if device_id is not None:
    list_args[6] = device_id

# 修改后
if device_id is not None and len(list_args) > 6:
    list_args[6] = device_id
```

### 2. scripts/data/add_agent_name.py (行 58-59)

从 train/val split 生成 Agent Loop 用的数据。新增字段：

- `agent_name`: "tool_agent" — verl 使用此字段选择 ToolAgentLoop
- `data_source`: "taubench_retail" — naive reward manager 的必需字段，取原始 `source` 字段
- `reward_model`: `{"ground_truth": "retail", "style": "rule"}` — naive reward manager 要求

```bash
python scripts/data/add_agent_name.py data/tau_bench/train_split_66.jsonl data/tau_bench/train_agent_66.jsonl
python scripts/data/add_agent_name.py data/tau_bench/val_split_18.jsonl data/tau_bench/val_agent_18.jsonl
```

### 3. verl-main-0516/verl/trainer/config/grpo_retail.yaml

完整训练配置（从 `scripts/train/grpo_retail.yaml` 复制）。关键参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `data.max_prompt_length` | 4096 | prompt ~2700 tokens with tool defs |
| `data.max_response_length` | 8192 | 多轮工具调用需要空间 |
| `data.truncation` | left | 超长 prompt 左截断 |
| `data.train_batch_size` | 2 | |
| `rollout.n` | 4 | 每 prompt 生成 4 条 |
| `rollout.prompt_length` | 4096 | **必须与 data.max_prompt_length 一致** |
| `rollout.response_length` | 8192 | **必须与 data.max_response_length 一致** |
| `rollout.gpu_memory_utilization` | 0.50 | 2B 够用，4B 需更低 |
| `actor.lora_rank` | 64 | |
| `actor.lora_alpha` | 128 | |
| `actor.target_modules` | [out_proj, down_proj] | 只选未 fused 的层 |
| `algorithm.adv_estimator` | grpo | 无需 critic |

**注意**: `data.max_prompt_length` 和 `rollout.prompt_length` 必须一致，否则会导致 prompt 截断不一致。

### 4. (未应用) truncation patches in agent_loop.py

**注意**: 以下 patch 在远程机器上应用了，但**本地未应用**。原因是远程和本地的 verl 版本不同：

- 远程 verl (v0.9.0.dev latest): 有 `_pad_token_ids()` 方法，有内置 `_post_init_prompt` prompt 截断
- 本地 verl-main-0516: 没有这些方法，使用 `tokenizer.pad()` 直接操作

远程应用的 patch 在 `_agent_loop_postprocess` 中：
```python
# Patch 1: Prompt pre-truncation (verl 内置已有，冗余)
prompt_ids = output.prompt_ids[-self.rollout_config.prompt_length:]

# Patch 2: Response pre-truncation (安全上限)
response_ids = output.response_ids[:self.rollout_config.response_length]

# Patch 3: Response mask pre-truncation
response_mask_data = output.response_mask[:self.rollout_config.response_length]

# Patch 4: 使用截断后的长度
start_pos = prompt_output["input_ids"].shape[1] - len(prompt_ids)
prompt_length=len(prompt_ids), response_length=len(response_ids),
```

如果需要安全上限（防止模型生成超长 response 导致 tensor 长度不一致），用 `tokenizer.pad()` 实现。

---

## 数据说明

### 数据来源

tau-bench retail — 客服对话任务。用户扮演客户（模拟），agent 扮演客服。涉及订单查询、退货、换货、退款、地址修改等零售场景。

### 数据流转

```
tau-bench 定义 (164 retail tasks)
    │
    ▼
┌─────────────────────────────────────────────┐
│ scripts/data/ 处理                           │
│                                              │
│ 1. convert_tau_bench.py  → 原始 JSONL        │
│ 2. filter_and_split.py    → train/val/test   │
│ 3. augment_data.py        → 每 task 多 variant│
│ 4. add_agent_name.py      → Agent Loop 格式   │
└─────────────────────────────────────────────┘
    │
    ▼
train_agent_66.jsonl  (251 条, 66 个 unique tasks, ~3.8 条/task)
val_agent_18.jsonl    ( 69 条, 18 个 unique tasks, ~3.8 条/task)
test_agent_37.jsonl   ( 37 条, 23 个 unique tasks, 用于 baseline)
```

### 数据隔离

train/val/test 按 **task_id** 分，同一 task 的所有 variant 只在其中一个 split，保证无泄漏。

### 每条数据格式

```json
{
    "id": "retail_task_12_v0",
    "source": "taubench_retail",
    "data_source": "taubench_retail",
    "task_id": "retail_task_12",
    "domain": "retail",
    "variant": "v0",
    "evaluation": {"rubric": "...", "expected_outcome": "..."},
    "tools": ["lookup_user", "lookup_order", ...],
    "prompt": [
        {"role": "system", "content": "You are a retail customer service agent..."},
        {"role": "user", "content": "Hi, I need help with my order..."}
    ],
    "agent_name": "tool_agent",
    "reward_model": {"ground_truth": "retail", "style": "rule"}
}
```

**关键字段**:
- `agent_name`: "tool_agent" → verl 使用 ToolAgentLoop 处理多轮工具调用
- `data_source`: na4ve reward manager 使用此字段分组
- `reward_model.ground_truth`: reward manager 根据 domain ("retail") 选择对应的 rubric engine
- `prompt`: messages 格式（非纯文本），verl 内部用 `apply_chat_template` + tools 生成 token ids

### 工具定义

34 个工具: 18 tau-bench 零售工具 + shared tools (think, transfer_to_human 等)。
在 `trainable_openclaw/training/agent_tools.py` 中用 `@function_tool` 装饰器注册。
verl Agent Loop 在 rollout 时加载并按 `qwen3_coder` XML 格式注入 prompt。

---

## Baseline 测试流程

### 架构

```
┌──────────────┐      ┌──────────┐      ┌─────────────────┐
│  vllm server │ ←──→ │ nanobot  │ ←──→ │ DeepSeek (user) │
│  Qwen3.5-4B  │      │ :8900    │      │ 模拟客户对话      │
│  port 8000   │      │          │      │ 评估任务完成度    │
└──────────────┘      └──────────┘      └─────────────────┘
```

- **vllm**: 提供 LLM 推理，max-model-len 49152，qwen3_coder tool parser
- **nanobot**: agent 框架，管理多轮对话、工具调用、上下文
- **DeepSeek-chat**: 模拟用户，按 task 定义的角色和场景与 agent 对话
- **评测指标**: completion rate（任务完成率）、avg rounds、first try rate、recovery rate

### 运行 baseline

```bash
# 1. 启动 vllm (远程)
bash scripts/deploy/start_experience.sh

# 2. 运行零售评测
python scripts/eval/run_full_eval.py \
    --model qwen3.5-4b \
    --domain retail \
    --test-file data/tau_bench/test_prompts_augmented.jsonl \
    --output evaluation_results/
```

### Baseline 成绩

| Config | Completion | 说明 |
|--------|-----------|------|
| Qwen3.5-2B | **0%** | 完全不会用工具 |
| Qwen3.5-4B, hermes, 12K ctx | 6.67% (2/30) | 旧 baseline (2026-06-13) |
| **Qwen3.5-4B, qwen3_coder, 48K** | **45.95% (17/37)** | 当前 baseline (2026-06-14) |

**2B 的问题**: benchmark 上 0% completion rate。模型输出 tool call token，但 nanobot 无法正确解析——格式不对，工具名或参数提取错误。这说明 2B 在没有额外训练的情况下不具备可靠的 tool-calling 能力。

这解释了 GRPO 训练时 `split_sizes=[4,4,4,4,4,4,4,4]` 错误的根因：2B 输出几近 EOS，无有用轨迹。

**潜在的解决方向**:
1. **先做 SFT 预热**: 用 4B 在 nanobot 上跑的 successful trajectories 做 dataset，SFT 2B 学习基本 tool-calling 格式
2. **换 4B + 更激进的 memory 策略**: 减小 KV cache、降低 batch、用 CPU offload
3. **从头训练一个 tool-calling adapter**: 用 2B + adapter 专门学习工具调用

### Baseline vs GRPO 的关系

- **Baseline**: static eval — 用 nanobot + vllm 跑对话，看 agent 能否完成任务
- **GRPO 训练**: dynamic training — verl 内部跑 Agent Loop rollout，用 reward 信号更新 LoRA 权重

两者共享 `agent_tools.py`（工具定义）和 reward rubrics。reward 框架见 `trainable_openclaw/training/grpo_reward.py`。

## 遇到的错误

### Error 1: IndexError in rebuild_ipc (PyTorch 2.10)
```
IndexError: list assignment index out of range
```
**根因**: PyTorch 2.10 IPC handle 格式变化。**已修复** (改动 #1)。

### Error 2: KeyError: 'data_source' 
**根因**: naive reward manager 需要此字段。**已修复** (改动 #3)。

### Error 3: KeyError: 'reward_model'
**根因**: naive reward manager 需要 `reward_model.ground_truth`。**已修复** (改动 #3)。

### Error 4: CUDA OOM (4B model)
```
vLLM cumem_allocator OOM during wake_up cycle
```
**根因**: 4B 模型 + LoRA + KV cache + actor optimizer 超过 48GB。降低 `gpu_memory_utilization` 到 0.25 反而导致 "No available memory for cache blocks"。
**解决**: 切换到 Qwen3.5-2B。

### Error 5: RuntimeError — split_sizes=[4,4,4,4,4,4,4,4] (未解决)
```
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 8192,
got split_sizes=[4,4,4,4,4,4,4,4]
```

完整调用链:
```
prepare_micro_batches → rearrange_micro_batches → index_select_tensor_dict
→ tensor.unbind() → torch.split(values, lengths_scalars)
```

每条序列只有 4 个有效 token（response_mask 中值为 1 的位置），远少于 8192。

**可能原因**:
1. ~~prompt 被截断到 1024~~ → 已修复 (prompt_length: 4096)，但仍报错
2. 2B 模型不理解多轮 tool-calling 格式，直接输出 EOS
3. chat template 或 tool definitions 格式有问题
4. tokenizer 行为异常

**建议调试方向**:
1. 单独用 vllm 加载 2B，发一条带 tool defs 的 prompt，看实际输出
2. 检查 `output.response_ids` 和 `output.response_mask` 的原始内容
3. 对比 4B baseline eval 时是否也有类似问题（4B eval 是直接调 nanobot 不是 verl）
4. 考虑先做 SFT 预热，让 2B 学会基本的 tool-calling 格式

## 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/train/grpo_retail.yaml` | 训练配置 (主) |
| `verl-main-0516/verl/trainer/config/grpo_retail.yaml` | 训练配置 (verl Hydra 用) |
| `ai_scripts/launch_training.sh` | 远程启动脚本 |
| `ai_scripts/remote_diag.py` | 环境诊断脚本 |
| `trainable_openclaw/training/agent_tools.py` | @function_tool 装饰器注册 |
| `trainable_openclaw/training/grpo_reward.py` | compute_score 函数 |
| `scripts/data/add_agent_name.py` | 数据预处理脚本 |
| `data/tau_bench/train_agent_66.jsonl` | 训练数据 (66条) |
| `data/tau_bench/val_agent_18.jsonl` | 验证数据 (18条) |

## 数据同步

使用 `scripts/tools/autodl_sync.py` 同步代码到远程：

```bash
python scripts/tools/autodl_sync.py --full           # 全量同步
python scripts/tools/autodl_sync.py --exec "CMD"     # 远程执行命令
python scripts/tools/autodl_sync.py --tail FILE      # 查看远程日志
python scripts/tools/autodl_sync.py --download REMOTE LOCAL  # 下载文件
```
