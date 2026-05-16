# 📚 LLM-as-Judge / 自动评估 — 论文参考

> 整理日期: 2026-05-16 | 项目: trainable-openclaw
>
> 🏆 = 顶会发表 | 📊 = 高引 |

---

## 🏆📊 顶会高引基石论文

### 1. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
- **arXiv**: [2306.05685](https://arxiv.org/abs/2306.05685)
- **作者**: Lianmin Zheng, Wei-Lin Chiang, Ying Sheng 等 (UC Berkeley / LMSYS)
- **🏆 NeurIPS 2024** | 📊 引用量: ~2000+
- **关键思想**: LLM-as-Judge 基准测试。提出 MT-Bench 和 Chatbot Arena，验证 LLM judge 与人类判断的一致性。
- **关键 Bias 发现**:
  - **Position bias**: 位置靠前更容易得高分
  - **Verbosity bias**: 更长回答更容易得高分  
  - **Self-enhancement bias**: 模型给自己生成的回答打更高分
- **与项目关系**: ⭐⭐⭐⭐⭐ 经典必读。Rubric 执行器必须处理这些 bias。

### 2. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment
- **arXiv**: [2303.16634](https://arxiv.org/abs/2303.16634)
- **作者**: Yang Liu 等 (微软)
- **🏆 EMNLP 2023** | 📊 引用量: ~800+
- **关键思想**: Chain-of-thought + form filling 范式让 LLM 做 NLG 评估，显著提升与人类一致性。
- **与项目关系**: ⭐⭐⭐⭐ CoT 评估范式可融入 Rubric 执行器设计。

### 3. Constitutional AI: Harmlessness from AI Feedback
- **arXiv**: [2212.08073](https://arxiv.org/abs/2212.08073)
- **作者**: Yuntao Bai 等 (Anthropic)
- **📊 引用量: ~1500+** | 极其有影响力
- **关键思想**: 用 constitution（原则列表）让 AI 自我批判和修正回复。
- **与项目关系**: ⭐⭐⭐⭐ Constitution ≈ Rubric。我们的 Rubric 是动态 CAI。

---

## 🔥 最新前沿（2026.05）

### 4. ODRPO: Ordinal Decompositions of Discrete Rewards for Robust Policy Optimization
- **arXiv**: [2605.12667](https://arxiv.org/abs/2605.12667)
- **作者**: Nirmal Patel, Fei Wang, Inderjit Dhillon (UT Austin / Google)
- **日期**: 2026-05-12
- **关键思想**: 
  - Auto-rater 评分有噪（prompt 敏感 + 采样随机性）→ 噪声污染 GRPO advantage
  - 将离散 rubrics 分解为 ordinal binary indicators 序列
  - 独立计算各阈值 advantage 并累加，防止 outlier 污染全局更新
  - **零额外计算开销**，Qwen2.5-7B +14.8% FACTS, +7.5% AlpacaEval
- **与项目关系**: ⭐⭐⭐⭐⭐ **直接解决 Rubric 评分噪声问题！**

### 5. Quantifying the Statistical Effect of Rubric Modifications on Human-Autorater Agreement
- **arXiv**: [2605.06283](https://arxiv.org/abs/2605.06283)
- **作者**: Jessica Huynh, Alfredo Gomez 等 (Google)
- **日期**: 2026-05-07
- **关键发现**:
  - ✅ 提供代表性示例 + 上下文 → 增加 human-autorater 一致性
  - ✅ 减少 positional bias → 增加一致性
  - ❌ 更高 rubric 复杂度 + 保守聚合 → 降低一致性
  - **结论**: 需针对 domain 和 rubric 做定制化分析
- **与项目关系**: ⭐⭐⭐⭐⭐ **直接指导 Rubric 设计原则。** 要提供示例、减少位置偏差、避免过度复杂。

### 6. Comparing Developer and LLM Biases in Code Evaluation
- **arXiv**: [2603.24586](https://arxiv.org/abs/2603.24586)
- **作者**: Aditya Mittal 等 (CMU)
- **日期**: 2026-03
- **与项目关系**: ⭐⭐⭐ 了解 LLM Judge 常见 bias。

---

## LLM Judge 必须处理的 Bias 清单

| Bias 类型 | 描述 | 缓解方法 |
|-----------|------|---------|
| Position bias | 靠前回答得分更高 | 随机打乱顺序、多次采样取平均 |
| Verbosity bias | 更长回答得分更高 | Rubric 明确长度无关、length penalty |
| Self-enhancement | 模型偏好自己的回答 | 使用不同模型做 Judge |
| Order bias | 评分顺序影响结果 | 随机化 + 交叉验证 |
| Egocentric bias | 与自身风格相似得分高 | 多模型 ensemble |

---

## 补充参考

### AlphaVerus / External Verification
- **搜索建议**: `self-improving RL external verifier`
- **相关概念**: 用外部验证器（compiler、test suite）作为 oracle reward，避免 LLM Judge bias。
- **项目关系**: ⭐⭐⭐ 代码类任务用 sandbox 执行结果作为 hard reward。
