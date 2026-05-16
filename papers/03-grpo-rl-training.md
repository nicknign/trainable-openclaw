# 📚 GRPO / RL for LLM Training — 论文参考

> 整理日期: 2026-05-16 | 项目: trainable-openclaw
>
> 🏆 = 顶会发表 | 📊 = 高引 | 🔥 = 2026 最新

---

## 🏆📊 RLHF / RL for LLM 基石论文

### 1. Training Language Models to Follow Instructions (InstructGPT)
- **arXiv**: [2203.02155](https://arxiv.org/abs/2203.02155)
- **作者**: Long Ouyang, Jeff Wu 等 (OpenAI)
- **🏆 NeurIPS 2022** | 📊 引用量: ~5000+
- **关键思想**: RLHF 经典三步流程——SFT → Reward Model → PPO。奠定了整个 RLHF 领域的基础。
- **与项目关系**: ⭐⭐⭐⭐⭐ RLHF 圣经。理解我们的"Rubric→打分→GRPO 训练"在 RLHF 谱系中的位置。

### 2. Direct Preference Optimization (DPO)
- **arXiv**: [2305.18290](https://arxiv.org/abs/2305.18290)
- **作者**: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (Stanford)
- **🏆 NeurIPS 2023** | 📊 引用量: ~3000+
- **关键思想**: 绕过单独的 reward model，直接从偏好数据优化策略。简化 RLHF 流程。
- **与项目关系**: ⭐⭐⭐ 我们的 Rubric 打分本质是构造 preference signal。

### 3. RLAIF: Scaling RL from Human Feedback with AI Feedback
- **arXiv**: [2309.00267](https://arxiv.org/abs/2309.00267)
- **作者**: Harrison Lee 等 (Google)
- **🏆 ICML 2024** | 📊 引用量: ~500+
- **关键思想**: 用 LLM 替代人类标注做 RLHF，效果不亚于人类标注。AI feedback pipeline。
- **与项目关系**: ⭐⭐⭐⭐⭐ **我们的 Rubric + LLM Judge 就是 RLAIF 的一种实现。**

### 4. Proximal Policy Optimization Algorithms (PPO)
- **arXiv**: [1707.06347](https://arxiv.org/abs/1707.06347)
- **作者**: John Schulman 等 (OpenAI)
- **📊 引用量: ~15000+** | RL 领域圣经级论文
- **关键思想**: Clipped objective + value function + entropy bonus。GRPO 的基础。
- **与项目关系**: ⭐⭐⭐ GRPO 是 PPO 的变体，理解 PPO 才能理解 GRPO。

---

## 🏆🔥 GRPO 核心论文

### 5. DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO 首发)
- **arXiv**: [2402.03300](https://arxiv.org/abs/2402.03300)
- **作者**: Zhihong Shao, Peiyi Wang 等 (DeepSeek)
- **📊 引用量: ~800+** | **首次提出 GRPO 算法**
- **GRPO 公式核心**:
  - 对每个 prompt 生成 N 个回答，group 内做 advantage 归一化
  - advantage = (reward - group_mean) / group_std
  - PPO-style clipped objective，不需要单独 critic 网络
  - 用 outcome reward（结果对错）而非 process reward
- **与项目关系**: ⭐⭐⭐⭐⭐ **GRPO 理论基础。** 我们的 Rubric 打分就是 GRPO 的 reward 来源。

### 6. Demystifying GRPO: Its Policy Gradient is a U-Statistic
- **arXiv**: [2603.01162](https://arxiv.org/abs/2603.01162)
- **作者**: Hongyi Zhou, Kai Ye 等
- **日期**: 2026-03 | 5 pages, 53 figures
- **关键思想**: GRPO policy gradient 的统计学解释——本质是 U-Statistic。为 group size 选择、方差估计提供理论指导。
- **与项目关系**: ⭐⭐⭐ 理论指导 group size 和 advantage 估计的调参。

---

## 🔥 2026 最新 GRPO 改进

### 7. GEAR: Granularity-Adaptive Advantage Reweighting
- **arXiv**: [2605.11853](https://arxiv.org/abs/2605.11853)
- **作者**: Sijia Li, Yuchen Huang 等 (微软)
- **日期**: 2026-05-12
- **关键思想**: Self-distillation 产生 token/segment 级 divergence 信号 → 识别语义偏离锚点 → 自适应调整 local advantage 权重。GRPO +20%。
- **与项目关系**: ⭐⭐⭐⭐ 与 Rubric 多维度打分结合——不同维度给不同粒度赋权。

### 8. SDAR: Self-Distilled Agentic Reinforcement Learning
- **arXiv**: [2605.15155](https://arxiv.org/abs/2605.15155)
- **作者**: Zhengxi Lu 等 (浙大 + 蚂蚁)
- **日期**: 2026-05-14
- **关键思想**: OPSD 作为 gated 辅助目标 + GRPO。token-level 稠密信号经 sigmoid gate 调制。
- **与项目关系**: ⭐⭐⭐⭐ GRPO + 辅助信号的 recipe。

### 9. ODRPO: Ordinal Decompositions for Robust Policy Optimization
- **arXiv**: [2605.12667](https://arxiv.org/abs/2605.12667)
- **作者**: Nirmal Patel 等 (UT Austin / Google)
- **日期**: 2026-05-12
- **与项目关系**: ⭐⭐⭐⭐⭐ Rubric 打分 → GRPO 的鲁棒 reward 设计。

---

## RL for LLM 算法全景对比

| 算法 | 需要 Critic | 需要 Reward Model | 数据需求 | 顶级论文 | 引用 |
|------|------------|-------------------|---------|---------|------|
| **PPO** | ✅ | ✅ | pairwise 偏好 | Schulman 2017 | ~15000 |
| **DPO** | ❌ | ❌ | pairwise 偏好 | Rafailov NeurIPS'23 | ~3000 |
| **GRPO** | ❌ | ❌ | outcome reward | DeepSeekMath 2024 | ~800 |
| **KTO** | ❌ | ❌ | 单样本好坏 | Ethayarajh 2024 | ~300 |
| **SPIN** | ❌ | ❌ | 人类文本 | Chen ICML'24 | ~400 |
| **RLOO** | ❌ | ❌ | outcome reward | Ahmadian 2024 | ~200 |

**我们的选择: GRPO** — 天然适配多答案生成 + Rubric 打分。

---

## GRPO 训练关键参数参考

| 参数 | 建议值 | 说明 |
|------|--------|------|
| group size (N) | 4-16 | 太小方差大，太大显存放不下 |
| clip range (ε) | 0.2-0.3 | 建议 clip higher（参考 Agentic RL 论文） |
| KL penalty (β) | 0.01-0.1 | 防止偏离 SFT 太远 |
| learning rate | 1e-6 ~ 5e-6 | LoRA 可适当提高到 1e-4 |
| entropy coefficient | 0.01-0.05 | 防止过早收敛 |
| overlong penalty | -0.5 ~ -1.0 | 惩罚过长生成 |

---

## VeRL 框架参考

- **仓库**: [volcengine/verl](https://github.com/volcengine/verl)
- **论文**: HybridFlow (EuroSys 2025)
- **关键设计**: 3FS 解耦架构 / Sleep/Wake 双模 / CheckpointEngine 权重同步
- **我们的改造**: rollout 常驻 → 空闲切训练 → 权重同步 → 恢复推理

---

## 🆕 补充综述

### Reward Hacking in the Era of Large Models
- **arXiv**: [2604.13602](https://arxiv.org/abs/2604.13602)
- **作者**: Xiaohua Wang 等 (复旦大学) | 42 pages
- **日期**: 2026-04
- **与项目关系**: ⭐⭐⭐⭐ Rubric 自进化必须防止 reward hacking。
