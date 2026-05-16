# 📚 LLM-as-Judge / 自动评估 — 论文参考

> 整理日期: 2026-05-16 | 项目: trainable-openclaw

---

## 核心参考

### 1. ODRPO: Ordinal Decompositions of Discrete Rewards for Robust Policy Optimization
- **arXiv**: [2605.12667](https://arxiv.org/abs/2605.12667)
- **作者**: Nirmal Patel, Fei Wang, Inderjit Dhillon (UT Austin / 谷歌)
- **日期**: 2026-05-12（最新！）
- **关键思想**: 
  - Auto-rater 的评分本质上有噪（prompt 敏感 + 采样随机性），噪声会污染 GRPO advantage 估计
  - 将离散的 rubrics 分解为 ordinal binary indicators 序列
  - 独立计算各阈值的 advantage 并累加，防止 outlier 污染全局更新
  - Qwen2.5-7B +14.8% FACTS-grounding-v2, +7.5% AlpacaEval
  - **零额外计算开销**
- **与项目关系**: ⭐⭐⭐⭐⭐ **直接解决我们 Rubric 评分的噪声问题！** Rubric 生成 → 多维度打分 → 通过 ordinal decomposition 消除噪声 → GRPO 更新。

### 2. Quantifying the Statistical Effect of Rubric Modifications on Human-Autorater Agreement
- **arXiv**: [2605.06283](https://arxiv.org/abs/2605.06283)
- **作者**: Jessica Huynh, Alfredo Gomez 等 (Google)
- **日期**: 2026-05-07
- **关键发现**:
  - 提供代表性示例 + 上下文 → 增加 human-autorater 一致性
  - 减少 positional bias → 增加一致性
  - 更高 rubric 复杂度 + 保守聚合 → 降低一致性
  - **结论**: 需要针对 domain 和 rubric 做定制化分析
- **与项目关系**: ⭐⭐⭐⭐⭐ **直接指导 Rubric 设计原则**：要提供示例、减少位置偏差、避免过度复杂。

### 3. Comparing Developer and LLM Biases in Code Evaluation
- **arXiv**: [2603.24586](https://arxiv.org/abs/2603.24586)
- **作者**: Aditya Mittal, Ryan Shar 等 (CMU)
- **日期**: 2026-03
- **关键思想**: LLM-as-Judge 在代码评估中的 bias 分析，对比人类开发者 bias。
- **与项目关系**: ⭐⭐⭐ 了解 LLM Judge 的常见 bias，指导 Rubric 设计。

---

## 经典必读

### 4. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
- **arXiv**: [2306.05685](https://arxiv.org/abs/2306.05685)
- **作者**: Lianmin Zheng 等 (UC Berkeley / LMSYS)
- **发表**: NeurIPS 2024
- **关键思想**: LLM-as-Judge 的基准测试。提出 MT-Bench 和 Chatbot Arena，验证 LLM judge 与人类判断的一致性。
- **关键 Bias**:
  - **Position bias**: 位置靠前的回答更容易得高分
  - **Verbosity bias**: 更长的回答更容易得高分
  - **Self-enhancement bias**: 模型给自己生成的回答打更高分
- **项目关系**: ⭐⭐⭐⭐⭐ 经典必读，知道 LLM Judge 的坑在哪里。

### 5. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment
- **arXiv**: [2303.16634](https://arxiv.org/abs/2303.16634)
- **作者**: Yang Liu 等 (微软)
- **发表**: EMNLP 2023
- **关键思想**: 用 chain-of-thought + form filling 范式让 LLM 做 NLG 评估，显著提升与人类的一致性。
- **项目关系**: ⭐⭐⭐⭐ CoT 评估范式可以融入 Rubric 执行器设计。

### 6. Constitutional AI: Harmlessness from AI Feedback
- **arXiv**: [2212.08073](https://arxiv.org/abs/2212.08073)
- **作者**: Yuntao Bai 等 (Anthropic)
- **日期**: 2022-12
- **关键思想**: 用 constitution（原则列表）让 AI 自我批判和修正回复，再基于修正后的回复做 RL 训练。不需要人类标注有害内容。
- **项目关系**: ⭐⭐⭐⭐ Constitution ≈ Rubric。我们的 Rubric 是动态生成的 constitution。RLAIF 训练流程可直接参考。

---

## 补充参考

### 7. AlphaVerus: Self-Improving RL with External Verification
- **搜索建议**: `self-improving RL external verifier`
- **相关概念**: 用外部验证器（compiler、test suite、API checker）作为 oracle reward，避免 LLM Judge bias。
- **项目关系**: ⭐⭐⭐ 可以作为 Rubric 的补充——对代码类任务用 sandbox 执行结果作为 hard reward。

### 8. LLM Judge Bias 综述
- **搜索建议**: `LLM as judge bias survey position verbosity self-enhancement`
- **需关注的 bias 类型**:
  1. Position bias（位置偏差）
  2. Verbosity bias（长度偏差）
  3. Self-enhancement bias（自我增强偏差）
  4. Order bias（顺序偏差）
  5. Egocentric bias（自我中心偏差）
- **项目关系**: ⭐⭐⭐⭐ 在我们的 Rubric 执行器中必须处理这些 bias。
