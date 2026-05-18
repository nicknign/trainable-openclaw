# 🌳 Orchard: Open-Source Agentic Modeling Framework 分析

> 整理日期: 2026-05-18 | 项目: trainable-openclaw
> 来源：老板在飞书分享（来自 AI 技术新闻日报）⚠️ 与 OpenClaw/Claw 生态高度相关

---

## 基本信息

| 维度 | 详情 |
|------|------|
| **论文** | [arXiv:2605.15040](https://arxiv.org/abs/2605.15040) |
| **标题** | Orchard: An Open-Source Agentic Modeling Framework |
| **作者** | Baolin Peng, Wenlin Yao, Qianhui Wu, Hao Cheng, Xiao Yu, Rui Yang, Tao Ge, Alessandro Sordoni, Xingdi Yuan, Yelong Shen, Pengcheng He, Tong Zhang, Zhou Yu, Jianfeng Gao |
| **机构** | Microsoft |
| **提交** | 2026-05-14 |
| **领域** | cs.AI / cs.CL |

---

## 核心思路

将 LLM 转变为能通过规划、推理、工具使用和多轮环境交互解决复杂任务的自主 Agent。Orchard 是一个**开源、可扩展的 Agentic 建模框架**。

### 架构核心：Orchard Env

轻量级环境服务，提供可复用的沙箱生命周期管理原语，跨任务域、Agent harness 和 pipeline 阶段通用。

### 三个 Agentic Modeling Recipes

| Recipe | 目标 | 关键技术 | 成果 |
|--------|------|----------|------|
| **Orchard-SWE** | 代码 Agent | 107K 轨迹蒸馏 + Credit-Assignment SFT + Balanced Adaptive Rollout RL | SWE-bench Verified: SFT 64.3%, SFT+RL **67.5%**（同规模开源 SOTA） |
| **Orchard-GUI** | 视觉-语言计算机使用 Agent | 仅 0.4K 蒸馏轨迹 + 2.2K 开放任务 | WebVoyager 74.1%, Online-Mind2Web 67.0%, DeepShop 64.0%（最强开源模型） |
| **Orchard-Claw** | 个人助理 Agent | 仅 0.2K 合成任务 | Claw-Eval pass@3 59.6%, 配 ZeroClaw harness 达 **73.9%** |

---

## 🔗 与 OpenClaw 生态的关系

### ⚠️ 重要发现

Orchard-Claw 直接关联 OpenClaw 生态系统：

- **Claw-Eval**：Orchard 定义的评估基准
- **ZeroClaw harness**：Orchard-Claw 配对使用的 agent harness
- "Claw" 命名表明该工作在 OpenClaw 平台上进行训练和评估

这意味着 **Microsoft 正在 OpenClaw 平台上进行 Agent 训练研究**，并且产生了公开发表的论文。

---

## 与 trainable-openclaw 的关联

### 🎯 最强的方向验证

Orchard 是 trainable-openclaw **最直接的相关工作**：

| 维度 | Orchard | trainable-openclaw |
|------|---------|---------------------|
| **训练方式** | 离线 SFT + RL（蒸馏大模型轨迹） | 在线 RL（用户交互反馈） |
| **数据来源** | 合成数据 + 大模型蒸馏 | 真实用户交互信号 |
| **训练时机** | 离线批量训练 | 空闲时自动触发（sleep/wake） |
| **平台** | OpenClaw（作为评估/部署平台） | OpenClaw（作为推理目标 + 训练引擎） |
| **基础设施** | Orchard Env（独立环境层） | veRL 原生双模（引擎内聚） |

### 💡 关键差异（trainable-openclaw 的竞争壁垒）

1. **在线 vs 离线**：Orchard 依赖大模型蒸馏（MiniMax-M2.5、Qwen3.5-397B）做 SFT，再 RL；trainable-openclaw 直接从真实交互中学习——数据成本更低，个性化更强

2. **引擎层 vs 框架层**：Orchard 是框架（需要独立部署 Env + 训练 pipeline）；trainable-openclaw 是引擎（veRL 原生支持推理+训练切换）

3. **Credit-Assignment SFT** 值得借鉴：Orchard 提出的"从失败轨迹的生产性片段学习"的思路，可以融入 trainable-openclaw 的训练信号提取逻辑

### 🚀 战略意义

Orchard 论文的发表证明：
- **OpenClaw 平台已被顶级研究机构（Microsoft）采用**作为 Agent 训练/评估平台
- **Agent 训练是 2026 年的核心研究方向**，Microsoft 投入了完整团队
- **trainable-openclaw 的定位（在 OpenClaw 平台上做 Agent 训练）与业界最前沿方向一致**
- **差异化空间明确**：Orchard 做离线蒸馏，trainable-openclaw 做在线进化——可以互补

---

## 关键引用

```bibtex
@misc{peng2026orchard,
  title={Orchard: An Open-Source Agentic Modeling Framework},
  author={Baolin Peng and Wenlin Yao and Qianhui Wu and Hao Cheng and Xiao Yu and Rui Yang and Tao Ge and Alessandro Sordoni and Xingdi Yuan and Yelong Shen and Pengcheng He and Tong Zhang and Zhou Yu and Jianfeng Gao},
  year={2026},
  eprint={2605.15040},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```
