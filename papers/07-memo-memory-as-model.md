# 🧠 MeMo: Memory as a Model 分析

> 整理日期: 2026-05-18 | 项目: trainable-openclaw
> 来源：老板在飞书分享（来自 AI 技术新闻日报）

---

## 基本信息

| 维度 | 详情 |
|------|------|
| **论文** | [arXiv:2605.15156](https://arxiv.org/abs/2605.15156) |
| **标题** | MeMo: Memory as a Model |
| **作者** | Ryan Wei Heng Quek, Sanghyuk Lee, Alfred Wei Lun Leong, Arun Verma, Alok Prakash, Nancy F. Chen, Bryan Kian Hsiang Low, Daniela Rus, Armando Solar-Lezama |
| **机构** | NUS（新加坡国立大学）、MIT |
| **提交** | 2026-05-14 |
| **领域** | cs.CL / cs.AI / cs.LG |

---

## 核心思路

LLM 训练完成后参数冻结，但实际应用需要持续注入新的领域知识。MeMo 的方案是：**训练一个独立的"记忆模型"**来编码新知识，不碰 LLM 参数。

### 关键优势

| 特性 | 说明 |
|------|------|
| **跨文档关系建模** | 能捕捉复杂的跨文档关联 |
| **检索噪声鲁棒** | 对检索结果中的噪音不敏感 |
| **无灾难性遗忘** | LLM 参数不变，不会遗忘原有知识 |
| **即插即用** | 不需要访问 LLM 权重或 logits，支持开源和闭源模型 |
| **推理成本固定** | 检索成本与语料库大小无关 |

### 实验

在 BrowseComp-Plus、NarrativeQA、MuSiQue 三个基准上验证，相比现有方法表现强劲。

---

## 与 trainable-openclaw 的关联

### 💡 可借鉴的思路

1. **记忆模型与训练模型分离** — MeMo 的思路是"知识更新不碰主干模型"。trainable-openclaw 的 LoRA 微调也是类似思路（不碰基座权重，只训练 adapter），但解决的是不同问题：MeMo 解决知识注入，trainable-openclaw 解决行为优化

2. **即插即用的设计哲学** — MeMo 强调不依赖模型权重，这与 trainable-openclaw 的"引擎级改造"形成对比：一个走外部插件路线，一个走引擎原生路线

3. **对 trainable-openclaw 的潜在启发** — 如果 Agent 需要快速获取新领域知识（比如金融领域术语），MeMo 式的记忆模型可以作为 trainable-openclaw 的补充模块，避免每次新领域都要重新训练

### ⚡ 差异化

| 维度 | MeMo | trainable-openclaw |
|------|------|---------------------|
| **优化目标** | 知识（Knowledge） | 行为（Behavior via RL） |
| **更新方式** | 训练独立记忆模型 | LoRA 微调推理引擎 |
| **信号来源** | 文档/知识库 | 用户交互反馈 + rubric 评估 |

---

## 关键引用

```bibtex
@misc{quek2026memo,
  title={MeMo: Memory as a Model},
  author={Ryan Wei Heng Quek and Sanghyuk Lee and Alfred Wei Lun Leong and Arun Verma and Alok Prakash and Nancy F. Chen and Bryan Kian Hsiang Low and Daniela Rus and Armando Solar-Lezama},
  year={2026},
  eprint={2605.15156},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
