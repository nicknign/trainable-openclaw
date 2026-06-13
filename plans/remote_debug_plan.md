# 远程调试计划 — nanobot + Qwen3-4B 评测验证

> 日期: 2026-06-13 | 状态: 待机器上线

## 问题回顾

| 问题 | 发现 | 影响 |
|------|------|------|
| nanobot :8900 单条限制 | `Only a single user message is supported` | 多轮对话无法用 nanobot API |
| vllm :8000 多轮可用 | vllm 原生 OpenAI API 支持多消息 | 应直连 vllm 而非经 nanobot |
| Qwen3-4B `<think>` 块 | 回复前输出 `<think>...</think>` (token 151667/151668 写死在训练中) | 可能破坏 tool call JSON 格式 |
| AgentRunner 上下文累积 | 多轮下 messages 列表增长 | 需 vllm 原生支持 |

## 模型策略调整

**方案 B（推荐）**: 换用 Qwen3.5 系列模型，think 模式可控。
- Qwen3-4B think 无法在模型层面关闭，`enable_thinking: false` 只改 prompt 模板
- Qwen3.5 系列对 tool calling 支持更好
- **需要确认**: 模型名称、vllm 版本要求

**方案 A（备选）**: 继续用 Qwen3-4B，应用层 strip_think + tool call 后处理。

## 架构调整

```
之前（有问题）：
  AgentRunner → nanobot :8900 → vllm :8000
  问题: nanobot 不支持多轮

之后（修正）：
  AgentRunner → vllm :8000 (OpenAI-compatible API 直连)
  SimulatedUser → deepseek-chat (不变)
```

nanobot 仍保留在 :8900 作 Gateway/WebUI，但评测 pipeline 不经过它。

## Step 1: 环境检查

机器上线后，用 `scripts/remote_check_env.py` 验证：

```bash
python scripts/autodl_sync.py  # 同步最新代码
python scripts/autodl_sync.py --exec "/data/anaconda3/bin/python /data/wangye/trainable-openclaw/scripts/remote_check_env.py"
```

检查项：
- [ ] vllm 0.18.1 可 import
- [ ] torch CUDA 可用
- [ ] Qwen3-4B 模型存在 `/data/models/Qwen3-4B`
- [ ] GPU RTX 4090 空闲
- [ ] nanobot 可 import (PYTHONPATH)

## Step 2: 启动 vllm + nanobot

执行 `scripts/start_experience.sh`：

```bash
python scripts/autodl_sync.py --exec "bash /data/wangye/trainable-openclaw/scripts/start_experience.sh"
```

验证：
- [ ] vllm :8000 `/v1/models` 返回模型列表
- [ ] vllm :8000 `/v1/chat/completions` 支持多轮消息
- [ ] nanobot :8900 `/health` 返回 OK
- [ ] nanobot 多轮对话：换模型后重新测试 nanobot API 是否解除单条限制
- [ ] GPU 显存占用 ~36GB

## Step 3: 升级 vllm + 下载新模型（如换模型）

```
# 升级 vllm（版本待确认）
/data/anaconda3/bin/pip install vllm==<NEW_VERSION>

# 下载 Qwen3.5 系列模型（如果需要新模型）
# pip install modelscope && modelscope download ...
```

## Step 4: 验证 vllm 工具调用

用新模型/关闭 think 的 Qwen3-4B 测试工具调用。
1. 给 vllm 发送带 tool 定义的请求
2. 检查返回的 tool_calls 是否能正确解析
3. 如果 `<think>` 干扰 → 需要后处理 strip_think

```python
# 测试: 发送需要工具调用的消息
tools = [{"type": "function", "function": {"name": "get_weather", ...}}]
messages = [{"role": "user", "content": "What's the weather?"}]
```

- [ ] vllm 返回合法 tool_calls
- [ ] `<think>` 块在 tool_calls 之前还是之中
- [ ] 确认 OpenAI SDK 能否解析响应

## Step 4: 修改 AgentRunner 适配 Qwen3-4B

`trainable_openclaw/evaluation/interactive_eval.py` 的 AgentRunner 当前假定标准 OpenAI 响应格式。可能需要的修改：

### 4a. strip_think 处理
如果 `<think>` 出现在文本响应中，影响 SimulatedUser 判断：
```python
import re
def strip_think(text: str) -> str:
    return re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
```

### 4b. tool_calls 解析兼容
如果 `<think>` 出现在 tool call 响应中，检查 OpenAI SDK 是否自动处理。如果不能，需要在 AgentRunner 中预处理。

### 4c. 降低 thinking 强度
尝试在请求中设置 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` 看是否能抑制 thinking。

- [ ] AgentRunner 能正确调用工具
- [ ] 文本响应不含 `<think>` 残留

## Step 5: 跑 1 通评测

修改后的 AgentRunner + vllm :8000 + SimulatedUser(deepseek-chat)，跑第 1 条 task。

```bash
python scripts/autodl_sync.py --exec "/data/anaconda3/bin/python /data/wangye/trainable-openclaw/scripts/remote_eval_one.py"
```

成功标准：
- [ ] Agent 收到用户消息后调用至少 1 次工具
- [ ] SimulatedUser 正确判断 agent 回复
- [ ] 对话在 ≤10 轮内完成（completed=True）
- [ ] Trajectory 日志完整可读

## 产出

```
scripts/
├── test_vllm_tools.py         # NEW: vllm 工具调用测试
├── remote_eval_one.py          # MODIFY: AGENT_BASE_URL → :8000
└── test_vllm_multi.py          # 已有: 多轮对话测试

trainable_openclaw/evaluation/
└── interactive_eval.py         # MAYBE MODIFY: strip_think, tool call compat
```
