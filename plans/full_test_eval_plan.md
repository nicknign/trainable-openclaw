# Full Test Set Evaluation Plan

> 日期: 2026-06-13 | 状态: 待执行 | 目标: 4B baseline 评分

## 目标

对 Qwen3.5-4B (nanobot) 在 tau-bench 测试集（50 prompt）上运行完整评测，产出 baseline 评分。

## 测试环境

| 组件 | 配置 |
|------|------|
| Agent 模型 | Qwen3.5-4B (vllm :8000, max-model-len 49152) |
| Agent 框架 | nanobot :8900 (context 49152, timeout 600s) |
| 模拟用户 | DeepSeek-chat (teacher mode, 10 轮上限) |
| 测试集 | data/tau_bench/test_prompts_augmented.jsonl (50 tasks) |

## 评分标准

**5 维评分体系 (0-1):**

| 指标 | 权重 | 计算方式 | 含义 |
|------|------|---------|------|
| Completion Rate | 40% | completed_tasks / total | 任务完成率 |
| Avg Satisfaction | 30% | mean(satisfaction) all tasks | 平均满意度（含失败） |
| First-Try Rate | 10% | tasks with rounds≤3 / total | 高效完成率 |
| Reliability | 10% | 1.0 - (agent_error_count / total) | 系统可靠性 |
| Persistence | 10% | 1.0 - (give_up_count / total) | 用户未放弃率 |

**总分 =** `0.4×completion + 0.3×satisfaction + 0.1×first_try + 0.1×reliability + 0.1×persistence`

**等级:**
- ≥0.8: 可上线
- 0.5-0.8: 可用需监督
- <0.5: 需显著改进

## 执行脚本

已有 `ai_scripts/batch_eval_runner.py` 支持：
- 逐任务执行，unbuffered 输出到 `/tmp/batch_eval.log`
- 结果写入 `/tmp/batch_eval_results.json`
- 每个 task 记录: completed, rounds, satisfaction, status, time

需修改：从 5 task 扩展到 50 task。

## 执行步骤

1. 确认远程服务健康（vllm :8000, nanobot :8900）
2. 修改 batch_eval_runner.py 的 range(5) → range(50)（或全部 task）
3. 同步到远程
4. 启动 nohup 后台执行（预计 2-3 小时）
5. 脚本完成后 `/tmp/batch_eval_results.json` 即最终结果
6. 运行 `python scripts/eval/run_full_eval.py` 计算评分（或从 JSON 直接计算）

## 验证标准

- [ ] 50 tasks 全部执行完成
- [ ] 无 agent_error（确保服务稳定）
- [ ] 产出 per-domain 评分（airline vs retail）
- [ ] 产出 5 维评分表
- [ ] 结果写入 `evaluation_results/baseline_4b.json`
