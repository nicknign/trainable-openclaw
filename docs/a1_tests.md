# A1: Rollout API Server — 测试文档

## 测试概述

A1 的测试位于 `tests/test_serve_ppo.py`，是一套 **无 GPU 依赖的冒烟测试**。通过 mock `LLMServerClient` 和 `tokenizer`，在本地即可验证完整 API 链路。

## 运行方式

```bash
cd projects/trainable-openclaw
python -m pytest tests/test_serve_ppo.py -v
```

## 测试覆盖矩阵

| # | 测试方法 | 类型 | 验证内容 | 对应代码行 |
|---|---------|------|---------|-----------|
| 1 | `test_chat_message` | 单元 | `ChatMessage` 模型字段和 `model_dump()` | serve_ppo.py:65-67 |
| 2 | `test_chat_completion_request` | 单元 | `ChatCompletionRequest` 字段赋值 + 默认值 (stream=False, top_p=None) | serve_ppo.py:70-77 |
| 3 | `test_chat_completion_request_defaults` | 单元 | 省略 model/temperature 时的默认值 (model="default", temperature=None) | serve_ppo.py:71,73 |
| 4 | `test_chat_completion_response` | 单元 | `ChatCompletionResponse` 完整结构：choices嵌套、usage统计、object字段 | serve_ppo.py:92-98 |
| 5 | `test_health_response` | 单元 | `HealthResponse` 模型字段 | serve_ppo.py:101-106 |
| 6 | `test_health` | 集成 | `GET /v1/health` 返回200 + status/mode/gpu_count | serve_ppo.py:131-139 |
| 7 | `test_chat_completions` | 集成 | `POST /v1/chat/completions` 返回完整 OpenAI 兼容响应 | serve_ppo.py:141-207 |
| 8 | `test_chat_completions_uses_default_sampling` | 集成 | 请求不含 sampling params 时回退到 rollout_config 默认值 | serve_ppo.py:162-165 |
| 9 | `test_chat_completions_uninitialized` | 集成 | 服务未初始化时返回 503 | serve_ppo.py:147-148 |

## Mock 依赖说明

测试通过 3 个 fixture 替换真实 GPU 依赖：

```
test_app(mock_tokenizer, mock_llm_client)
├── mock_tokenizer              → MagicMock，模拟 tokenizer.apply_chat_template / encode / decode
├── mock_llm_client             → AsyncMock，模拟 LLMServerClient.generate() 返回固定 TokenOutput
└── _app_state (模块级字典)     → 注入 rollout_config / gpu_count / active_requests / start_time
```

关键注入点（`test_serve_ppo.py:131-138`）：
```python
_app_state["llm_client"] = mock_llm_client
_app_state["tokenizer"] = mock_tokenizer
_app_state["rollout_config"] = {"temperature": 0.7, "top_p": 1.0, "top_k": -1, "response_length": 2048}
_app_state["gpu_count"] = 4
_app_state["active_requests"] = 0
_app_state["start_time"] = 1000.0
```

## 验证的核心行为

### 1. OpenAI 兼容性
- 请求格式：`{"model": "test", "messages": [{"role": "user", "content": "Hello"}], ...}`
- 响应格式：`{"object": "chat.completion", "choices": [...], "usage": {...}}`
- 所有字段名和下划线风格与 OpenAI API 对齐

### 2. 采样参数回退逻辑
- 请求提供 `temperature` → 使用请求值
- 请求不提供 `temperature` → 使用 `rollout_config` 默认值 (0.7)
- 同理 `top_p`、`top_k`、`max_tokens`

### 3. 错误处理
- 服务未初始化（`llm_client is None`）→ 503 Service Unavailable
- 生成异常 → 500 Internal Server Error
- `active_requests` 通过 `try/finally` 保证正确递减

### 4. Token 统计
- `prompt_tokens` = len(prompt_ids)
- `completion_tokens` = len(output.token_ids)
- `total_tokens` = prompt_tokens + completion_tokens

## 未覆盖项（需 Linux + GPU 环境验证）

| 场景 | 验证方式 |
|------|---------|
| 服务持续运行不退出 | 启动后等待 + 多次 curl |
| 多次请求独立采样结果不同 | 同一 prompt 两次请求，输出不同 |
| TP/PP 多卡并行推理 | 多 GPU 环境启动 |
| vLLM/SGLang 后端切换 | 切换 rollout.name 配置 |
| 真实模型生成质量 | 加载实际模型 |

## 测试数据流

```
TestClient.post("/v1/chat/completions", json=...)
  └─> fastapi endpoint chat_completions(req: ChatCompletionRequest)
        ├─> mock_tokenizer.apply_chat_template(messages) → "<|user|>Hello<|assistant|>"
        ├─> mock_tokenizer.encode(prompt_text) → [1, 2, 3, 4, 5]
        ├─> mock_llm_client.generate(prompt_ids, sampling_params) → TokenOutput([10,20,30])
        └─> mock_tokenizer.decode([10,20,30]) → "Hello! This is a test response."
              └─> ChatCompletionResponse(id=..., choices=[...], usage=...)
```
