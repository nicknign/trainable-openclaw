# 🔧 开源实现参考与双模引擎现状分析

> 整理日期: 2026-05-16 | 项目: trainable-openclaw

---

## 已有开源实现（可直接借鉴代码）

### 1. ⭐ Open-AgentRL (Gen-Verse) — 最直接参考
- **仓库**: https://github.com/Gen-Verse/Open-AgentRL
- **关联论文**: Demystifying RL in Agentic Reasoning + RLAnything (ICML 2026)
- **可借鉴部分**:
  - `recipe/` — GRPO 训练配方参数（clip higher、entropy、overlong penalty）
  - `reward/` — reward function 实现（outcome + step-wise 混合信号）
  - `configs/` — 多环境配置（ALFWorld, OSWorld, WebShop 等）
  - `data/` — 端到端 tool-use 轨迹数据处理
  - `scripts/` — 训练启动脚本
- **与我们的差异**: 他们是批量离线训练，我们是持续在线学习。recipe 可复用，架构不可复用。

### 2. SPIN (UCLA)
- **仓库**: https://github.com/uclaml/SPIN
- **关联论文**: SPIN (ICML 2024, ~400 引用)
- **可借鉴部分**:
  - Self-play 数据生成 pipeline
  - 区分"人类文本 vs 模型生成文本"的判别器设计
  - 迭代训练循环逻辑

### 3. veRL (字节跳动) — 核心引擎
- **仓库**: https://github.com/volcengine/verl
- **关键文件**: `verl/trainer/ppo/ray_trainer.py` → `fit()` 方法
- **sleep/wake 机制**（最接近我们的双模引擎）:
  ```python
  # 伪代码示意
  def fit(self):
      for epoch in range(total_epochs):
          # Wake: rollout 生成
          gen_batch = self.rollout_wg.generate_sequences(prompts)
          
          # Sleep: 释放 GPU 显存
          self.actor_rollout_wg.sleep()
          
          # Train: 训练更新
          self.train(gen_batch)
          
          # Wake: 恢复推理
          self.actor_rollout_wg.wake()
          
          # 权重同步: CheckpointEngine
          self.ckpt_engine.update_weights()
  ```
- **我们改造的方向**: 把 `fit()` 的单次循环改成常驻服务——gen→gen→...→(空闲)→sleep→train→wake→gen
- **可直接复用的**: `sleep()`, `wake()`, `CheckpointEngine.update_weights()`, `AsyncActorRolloutRefWorker`

### 4. vLLM LoRA Adapter 热加载
- **仓库**: https://github.com/vllm-project/vllm
- **相关特性**: 支持运行时加载/卸载 LoRA adapter，不需要重启服务
- **可借鉴部分**: LoRA 权重热加载机制、并发推理接口
- **局限**: 只有推理没有训练，不是完整双模

### 5. Self-Rewarding LM (Meta)
- **仓库**: https://github.com/facebookresearch/self-rewarding-lm (推测)
- **可借鉴部分**: LLM-as-Judge prompt 模板、DPO 迭代训练流程

---

## 核心问题：是否已有可更新权重的推理引擎原型？

### 答案：**没有现成的完全符合需求的实现。**

| 项目 | 推理服务 | 训练更新 | 权重同步 | 空闲触发 | HTTP API |
|------|---------|---------|---------|---------|---------|
| **veRL** | ✅ (batch) | ✅ | ✅ | ❌ | ❌ |
| **vLLM** | ✅ (streaming) | ❌ | ✅ (LoRA hot-swap) | ❌ | ✅ |
| **Open-AgentRL** | ❌ | ✅ (GRPO) | ❌ | ❌ | ❌ |
| **SGLang** | ✅ (streaming) | ❌ | ❌ | ❌ | ✅ |
| **我们的目标** | ✅ (persistent) | ✅ (LoRA) | ✅ | ✅ | ✅ |

### 最近似的东西：veRL 的 sleep/wake

veRL 的 `fit()` 循环本质就是**双模引擎的简化版**：
1. rollout → 推理生成
2. sleep → 释放资源
3. train → 参数更新
4. wake → 恢复推理
5. CheckpointEngine.update_weights() → 权重同步

**但它是批处理模式，不是常驻服务**——这就是我们需要改造的核心。

### 我们的工作本质上是：
```
veRL 的 sleep/wake + vLLM 的 HTTP API + 空闲检测调度器
= 可自进化的推理服务引擎
```

**结论：trainable-openclaw 填补了一个真实存在的空白。** 不是重复造轮子，是把已有的轮子（veRL sleep/wake、vLLM LoRA hot-swap）以新的方式组合起来。
