# 📚 自进化 Agent / Self-Improving LLM — 论文参考

> 整理日期: 2026-05-16 | 项目: trainable-openclaw

---

## 核心参考（必读）

### 1. Self-Rewarding Language Models
- **arXiv**: [2401.10020](https://arxiv.org/abs/2401.10020)
- **作者**: Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston (Meta)
- **发表**: ICML 2024
- **关键思想**: LLM 自己当 reward model，通过 LLM-as-Judge 给自己生成的回复打分，用 DPO 迭代训练。每次迭代模型既能生成更好的回复，也能打出更准的分数。
- **与项目关系**: ⭐⭐⭐ 这是自进化闭环的经典基线。我们的 Rubric 自生成 + GRPO 打分可以看作这个思路的进阶版。

### 2. Evolving-RL: End-to-End Optimization of Experience-Driven Self-Evolving Capability
- **arXiv**: [2605.10663](https://arxiv.org/abs/2605.10663)
- **作者**: Zhiyuan Fan, Wenwei Jin, Feng Zhang, Bin Li, Yihong Dong, Yao Hu, Jiawei Li
- **日期**: 2026-05-11（最新！）
- **关键思想**: 联合优化"经验提取"和"经验利用"两个能力。用两个监督信号分别训练 extractor 和 solver，实现协调共进化。ALFWorld unseen +98.7%，Mind2Web +35.8%（相对 GRPO baseline）。
- **与项目关系**: ⭐⭐⭐⭐⭐ 非常直接相关！经验驱动自进化 + RL 联合优化，本质和我们的"从对话反馈中学习"一样。区别在于他们用 experience memory，我们用 Rubric 评价。

### 3. Self-Distilled Agentic Reinforcement Learning (SDAR)
- **arXiv**: [2605.15155](https://arxiv.org/abs/2605.15155)
- **作者**: Zhengxi Lu, Zhiyuan Yao, Zhuowen Han 等 (浙江大学 + 蚂蚁集团)
- **日期**: 2026-05-14（最新！）
- **关键思想**: On-Policy Self-Distillation + GRPO 结合。teacher 分支提供 token-level 稠密信号，作为 GRPO 的辅助目标。gated 机制处理 teacher 负信号。ALFWorld +9.4%, Search-QA +7.0%, WebShop +10.2%。
- **与项目关系**: ⭐⭐⭐⭐ token-level 稠密信号 vs 我们的 Rubric 多答案打分互补。gated 蒸馏思路可以借鉴到 Rubric 打分中（低置信度 rubric 降低权重）。

### 4. Demystifying Reinforcement Learning in Agentic Reasoning
- **arXiv**: [2510.11701](https://arxiv.org/abs/2510.11701)
- **作者**: Zhaochen Yu, Ling Yang, Jiaru Zou, Shuicheng Yan, Mengdi Wang (Gen-Verse)
- **日期**: 2025-10
- **关键思想**: 从 data/algorithm/reasoning mode 三个维度系统研究 Agentic RL。4B > 32B。真实端到端工具调用轨迹 >> 合成轨迹。Clip higher + overlong reward shaping + 熵保持。
- **项目关系**: ⭐⭐⭐⭐⭐ 训练配方直接可用。recipe 设计思路（数据质量 > 模型大小）是我们项目的核心哲学。

### 5. RLAnything: Forge Environment, Policy, and Reward Model
- **arXiv**: [2602.02488](https://arxiv.org/abs/2602.02488)
- **作者**: Yinjie Wang, Tianbao Xie, Ke Shen, Mengdi Wang, Ling Yang (Gen-Verse)
- **日期**: 2026-02 | **ICML 2026**
- **关键思想**: 环境/策略/奖励三者闭环联合优化。环境自动适应（Critic Feedback）。Qwen3-VL-8B +9.1% OSWorld, Qwen2.5-7B +18.7% AlfWorld。
- **项目关系**: ⭐⭐⭐⭐ 环境自适应 + reward model co-optimization 可以启发我们的 Rubric 自演进机制。

---

## 扩展参考

### 6. Power Distribution Bridges Sampling, Self-Reward RL, and Self-Distillation
- **arXiv**: [2605.04542](https://arxiv.org/abs/2605.04542)
- **作者**: Akiyoshi Tomihari, Issei Sato (东京大学)
- **日期**: 2026-05-06
- **关键思想**: 分析采样分布、self-reward RL、self-distillation 之间的桥梁关系。
- **项目关系**: ⭐⭐⭐ 理论分析，帮助理解为什么 self-reward 机制有效。

### 7. Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning
- **arXiv**: [2601.08198](https://arxiv.org/abs/2601.08198)
- **作者**: Yibo Wang, Hai-Long Sun 等 (阿里巴巴)
- **发表**: NeurIPS 2025
- **关键思想**: 从 pairwise 升级到 triplet 比较的 self-play fine-tuning，更稳定。
- **项目关系**: ⭐⭐⭐ GRPO 天然多答案（>2），与 triplet 思路一致。

### 8. GEAR: Granularity-Adaptive Advantage Reweighting for LLM Agents
- **arXiv**: [2605.11853](https://arxiv.org/abs/2605.11853)
- **作者**: Sijia Li, Yuchen Huang 等 (微软)
- **日期**: 2026-05-12
- **关键思想**: 自适应粒度 advantage reweighting。用 self-distillation 产生的 divergence 信号识别语义偏离点，在 token/segment 级动态调整 advantage 权重。GRPO +20%。
- **项目关系**: ⭐⭐⭐⭐ advantage reweighting 可以改进 GRPO 训练效率，与我们的 Rubric 多维度打分结合。

### 9. GRPO Demystified: Its Policy Gradient is a U-Statistic
- **arXiv**: [2603.01162](https://arxiv.org/abs/2603.01162)
- **作者**: Hongyi Zhou, Kai Ye 等
- **日期**: 2026-03
- **关键思想**: GRPO policy gradient 的统计学解释——本质是 U-Statistic。
- **项目关系**: ⭐⭐⭐ 理论理解，帮助 debug GRPO 训练问题。

### 10. SPIN: Self-Play Fine-Tuning Converts Weak LLMs to Strong LLMs
- **arXiv**: [2401.01335](https://arxiv.org/abs/2401.01335)
- **作者**: Zixiang Chen 等 (UCLA)
- **发表**: ICML 2024
- **关键思想**: 模型和自己对弈，区分自己生成的和人类写的文本，逐渐提升生成质量。
- **项目关系**: ⭐⭐ 经典 self-play baseline。
