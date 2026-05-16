# 📚 GRPO / RL for LLM Training — 论文参考

> 整理日期: 2026-05-16 | 项目: trainable-openclaw

---

## 核心参考

### 1. DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO 原始论文)
- **arXiv**: [2402.03300](https://arxiv.org/abs/2402.03300)
- **作者**: Zhihong Shao, Peiyi Wang 等 (DeepSeek)
- **日期**: 2024-02
- **关键思想**: 
  - **首次提出 GRPO** (Group Relative Policy Optimization)
  - 对每个 prompt 生成 N 个回答，在 group 内做 advantage 归一化
  - 不需要单独的价值网络（critic），节省显存
  - 用 outcome reward（结果对错）而非 process reward
- **GRPO 公式核心**:
  - 对 group 内 G 个回答计算 reward
  - advantage = (reward - mean) / std
  - PPO-style clipped objective
- **与项目关系**: ⭐⭐⭐⭐⭐ **GRPO 理论基础。** 我们的 Rubric 打分就是 GRPO 的 reward 来源。

### 2. Demystifying Group Relative Policy Optimization: Its Policy Gradient is a U-Statistic
- **arXiv**: [2603.01162](https://arxiv.org/abs/2603.01162)
- **作者**: Hongyi Zhou, Kai Ye 等
- **日期**: 2026-03
- **关键思想**: 从统计学角度解释 GRPO policy gradient，证明其本质是 U-Statistic。为 group size 选择、方差估计提供理论指导。
- **与项目关系**: ⭐⭐⭐ 理论指导 group size 和 advantage 估计的调参。

### 3. GEAR: Granularity-Adaptive Advantage Reweighting
- **arXiv**: [2605.11853](https://arxiv.org/abs/2605.11853)
- **作者**: Sijia Li, Yuchen Huang 等 (微软)
- **日期**: 2026-05-12（最新！）
- **关键思想**:
  - 传统 GRPO 只用 outcome-level reward（粗粒度）
  - 通过 self-distillation 获得 token/segment 级 divergence 信号
  - divergence 尖峰处识别为语义偏离锚点 → 自适应调整 local advantage 权重
  - Qwen3 4B/8B，数学推理 + agentic tool-use，GRPO +20%
- **与项目关系**: ⭐⭐⭐⭐ **改进 GRPO advantage 计算。** 与 Rubric 多维度打分结合——不同 rubric 维度给不同粒度赋权。
- **代码**: 待关注是否开源

### 4. SDAR: Self-Distilled Agentic Reinforcement Learning
- **arXiv**: [2605.15155](https://arxiv.org/abs/2605.15155)
- **作者**: Zhengxi Lu 等 (浙大 + 蚂蚁)
- **日期**: 2026-05-14（最新！）
- **关键思想**: OPSD 作为 gated 辅助目标 + GRPO 主干。teacher 的 token-level 信号经 sigmoid gate 调制。ALFWorld +9.4%, WebShop +10.2%。
- **与项目关系**: ⭐⭐⭐⭐ GRPO + 辅助信号的 recipe 可直接参考。

### 5. ODRPO: Ordinal Decompositions for Robust Policy Optimization
- **arXiv**: [2605.12667](https://arxiv.org/abs/2605.12667)
- **作者**: Nirmal Patel 等 (UT Austin / Google)
- **日期**: 2026-05-12
- **关键思想**: 解决 auto-rater 噪声问题。将离散 rubrics 分解为 ordinal binary indicators，独立计算 advantage。
- **与项目关系**: ⭐⭐⭐⭐⭐ Rubric 打分 → GRPO 的鲁棒 reward 设计。

---

## RL for LLM 算法对比

| 算法 | 特点 | Critic | 适用场景 | 论文 |
|------|------|--------|---------|------|
| **GRPO** | group 内归一化 advantage | 不需要 | 多答案生成、数学/代码 | DeepSeekMath 2402.03300 |
| **PPO** | 经典 RL，需 critic | 需要 | 通用 RLHF | Schulman 2017 |
| **DPO** | 直接优化偏好，不用 RL | 不需要 | 偏好数据充足 | Rafailov 2023 |
| **KTO** | 不需要 pairwise 偏好 | 不需要 | 只有单一样本好坏信号 | Ethayarajh 2024 |
| **SPIN** | self-play fine-tuning | 不需要 | 迭代自我改进 | Chen 2024 |
| **RLOO** | REINFORCE Leave-One-Out | 不需要 | GRPO 的简化替代 | Ahmadian 2024 |

---

## GRPO 训练关键参数

| 参数 | 建议 | 说明 |
|------|------|------|
| group size (N) | 4-16 | 太小方差大，太大显存放不下 |
| clip range (ε) | 0.2 | PPO 默认值，GRPO 建议 clip higher |
| KL penalty (β) | 0.01-0.1 | 防止偏离 SFT 太远 |
| learning rate | 1e-6 ~ 5e-6 | LoRA 可适当提高 |
| entropy coefficient | 0.01-0.05 | 防止过早收敛 |
| overlong penalty | -0.5 ~ -1.0 | 惩罚过长生成 |

---

## 经典 RLHF/RLAIF 必读

### Proximal Policy Optimization Algorithms
- **arXiv**: [1707.06347](https://arxiv.org/abs/1707.06347)
- **作者**: John Schulman 等 (OpenAI)
- **日期**: 2017
- **关键思想**: PPO 原始论文——GRPO 的基础。clipped objective + value function + entropy bonus。

### Training Language Models to Follow Instructions (InstructGPT)
- **arXiv**: [2203.02155](https://arxiv.org/abs/2203.02155)
- **作者**: Long Ouyang 等 (OpenAI)
- **日期**: 2022
- **关键思想**: RLHF 经典流程——SFT → RM → PPO。人类偏好数据 + reward model。

### Direct Preference Optimization (DPO)
- **arXiv**: [2305.18290](https://arxiv.org/abs/2305.18290)
- **作者**: Rafael Rafailov 等 (Stanford)
- **日期**: 2023
- **关键思想**: 绕过 reward model，直接从偏好数据优化策略。简化 RLHF 流程。

### RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback
- **arXiv**: [2309.00267](https://arxiv.org/abs/2309.00267)
- **作者**: Harrison Lee 等 (Google)
- **日期**: 2023
- **关键思想**: 用 LLM 替代人类标注做 RLHF，效果不亚于人类标注。AI feedback > human feedback pipeline。
- **项目关系**: ⭐⭐⭐⭐⭐ 我们的 Rubric + LLM Judge 就是 RLAIF 的一种实现。

---

## VeRL 框架相关

- **仓库**: [volcengine/verl](https://github.com/volcengine/verl)
- **论文**: HybridFlow (EuroSys 2025)
- **关键设计**: 
  - 3FS: 解耦的控制流/数据流/计算流
  - Sleep/Wake 模式：rollout 唤醒 → 训练休眠
  - CheckpointEngine: 训练完自动同步权重 → rollout 用新权重
- **我们的改造点**: rollout 阶段常驻（不 sleep），空闲时才切训练。
