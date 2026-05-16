# 📚 自进化 Agent / Self-Improving LLM — 论文参考

> 整理日期: 2026-05-16 | 项目: trainable-openclaw
> 
> 🏆 = 顶会发表 | 📊 = 高引 |

---

## 🏆📊 顶会高引基石论文（必读）

### 1. Self-Rewarding Language Models
- **arXiv**: [2401.10020](https://arxiv.org/abs/2401.10020)
- **作者**: Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston (Meta FAIR)
- **🏆 ICML 2024** | 📊 引用量: ~500+
- **关键思想**: LLM 自己当 reward model，通过 LLM-as-Judge 给自己打分，用 DPO 迭代训练。每次迭代模型既能生成更好的回复，也能打出更准的分数。「要达到超人级 agent，需要超人级 feedback」。
- **与项目关系**: ⭐⭐⭐ 自进化闭环的经典基线。我们的 Rubric 自生成 + GRPO 打分是这个思路的进阶版。

### 2. SPIN: Self-Play Fine-Tuning Converts Weak LLMs to Strong LLMs
- **arXiv**: [2401.01335](https://arxiv.org/abs/2401.01335)
- **作者**: Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu (UCLA)
- **🏆 ICML 2024** | 📊 引用量: ~400+
- **关键思想**: 模型和自己对弈——区分自己生成的 vs 人类写的文本，逐步提升生成质量。不需要额外人类标注数据。
- **与项目关系**: ⭐⭐ Self-play 的经典范式，与我们 online learning 理念一致。

### 3. Constitutional AI: Harmlessness from AI Feedback
- **arXiv**: [2212.08073](https://arxiv.org/abs/2212.08073)
- **作者**: Yuntao Bai 等 (Anthropic)
- **🏆 未正式发表但极有影响力** | 📊 引用量: ~1500+
- **关键思想**: 用 constitution（原则列表）让 AI 自我批判和修正回复，再基于修正后回复做 RL 训练。不需要人类标注有害内容。
- **与项目关系**: ⭐⭐⭐⭐ Constitution ≈ Rubric。我们的 Rubric 是动态生成的 constitution。RLAIF 训练流程可直接参考。

### 4. DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO 原始论文)
- **arXiv**: [2402.03300](https://arxiv.org/abs/2402.03300)
- **作者**: Zhihong Shao, Peiyi Wang 等 (DeepSeek)
- **📊 引用量: ~800+** | 首次提出 GRPO 算法
- **与项目关系**: ⭐⭐⭐⭐⭐ GRPO 理论基础。我们的 Rubric 打分就是 GRPO 的 reward 来源。

---

## 🔥 最新前沿论文（2026.05，待顶会审核）

### 5. Evolving-RL: End-to-End Optimization of Experience-Driven Self-Evolving Capability
- **arXiv**: [2605.10663](https://arxiv.org/abs/2605.10663)
- **作者**: Zhiyuan Fan, Wenwei Jin, Feng Zhang 等
- **日期**: 2026-05-11 | 17 pages, 5 figures
- **关键思想**: 联合优化"经验提取"和"经验利用"。ALFWorld unseen +98.7% (vs GRPO)，Mind2Web +35.8%。
- **与项目关系**: ⭐⭐⭐⭐⭐ 非常直接相关！经验驱动自进化 + RL，本质和我们的"从反馈学习"一样。

### 6. SDAR: Self-Distilled Agentic Reinforcement Learning
- **arXiv**: [2605.15155](https://arxiv.org/abs/2605.15155)
- **作者**: Zhengxi Lu 等 (浙江大学 + 蚂蚁集团)
- **日期**: 2026-05-14
- **关键思想**: OPSD 作为 gated 辅助目标 + GRPO。ALFWorld +9.4%, WebShop +10.2%。
- **与项目关系**: ⭐⭐⭐⭐ token-level 稠密信号与 Rubric 多答案打分互补。

### 7. Demystifying RL in Agentic Reasoning
- **arXiv**: [2510.11701](https://arxiv.org/abs/2510.11701)
- **作者**: Zhaochen Yu, Ling Yang 等 (Gen-Verse)
- **日期**: 2025-10 | **代码**: [Open-AgentRL](https://github.com/Gen-Verse/Open-AgentRL)
- **关键思想**: 4B > 32B。真实端到端工具调用轨迹 >> 合成轨迹。Clip higher + entropy + overlong reward。
- **与项目关系**: ⭐⭐⭐⭐⭐ 训练配方直接可用。

### 8. RLAnything: Forge Environment, Policy, and Reward Model
- **arXiv**: [2602.02488](https://arxiv.org/abs/2602.02488)
- **作者**: Yinjie Wang 等 (Gen-Verse)
- **🏆 ICML 2026 Accepted**
- **关键思想**: 环境/策略/奖励三者闭环联合优化。Qwen3-VL-8B +9.1% OSWorld。
- **与项目关系**: ⭐⭐⭐⭐ 环境自适应 + reward model co-optimization 启发 Rubric 自演进。

---

## 扩展参考

### 9. Power Distribution Bridges Sampling, Self-Reward RL, and Self-Distillation
- **arXiv**: [2605.04542](https://arxiv.org/abs/2605.04542)
- **作者**: Akiyoshi Tomihari, Issei Sato (东京大学)
- **日期**: 2026-05-06
- **与项目关系**: ⭐⭐⭐ 理论分析 self-reward 机制有效性。

### 10. Triplets Better Than Pairs: Self-Play Fine-Tuning
- **arXiv**: [2601.08198](https://arxiv.org/abs/2601.08198)
- **作者**: Yibo Wang 等 (阿里巴巴)
- **🏆 NeurIPS 2025**
- **与项目关系**: ⭐⭐⭐ GRPO 天然多答案（>2），与 triplet 思路一致。

### 11. Reward Hacking in the Era of Large Models (综述)
- **arXiv**: [2604.13602](https://arxiv.org/abs/2604.13602)
- **作者**: Xiaohua Wang 等 (复旦大学)
- **日期**: 2026-04 | 42 pages
- **关键思想**: RLHF reward hacking 的全面综述——机制、涌现失配、挑战。
- **与项目关系**: ⭐⭐⭐⭐ 我们的 Rubric 评估必须防止 reward hacking。必读综述。
