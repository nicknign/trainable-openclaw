---
name: datasets-self-evolution-pipeline
description: Comprehensive catalog of datasets suitable for the trainable-openclaw self-evolution (correction→feedback→revision) training pipeline, including evaluation benchmarks
metadata:
  type: reference
---

# Datasets for Self-Evolution Training Pipeline

Research date: 2026-06-11

## Core requirement
Training data format: (prompt, incorrect_answer, correction_feedback, corrected_answer) tuples for GRPO training with rubric-based rewards.

## Top Recommendations (Ranked)

### 1. HelpSteer3 (NVIDIA, ACL 2025) -- BEST MATCH
- HuggingFace: `nvidia/HelpSteer3`
- License: CC-BY-4.0
- 40,821 feedback samples + edit/revision pairs
- Format: context (multi-turn) → 2 responses → detailed multi-sentence feedback → edited responses
- Feedback-Edit ITS pipeline: Draft → Feedback → Edit → Select
- Domain coverage: general (46%), STEM (12.2%), Code (21.9%), Multilingual (19.9%)
- Expert annotators (3-5 per sample), Cohen's κ > 0.8
- How to use: Feedback can be used as correction_feedback; Edit pairs as (answer, corrected_answer); multi-turn context as prompt
- Caveat: Feedback is about response quality, not user-initiated corrections. Needs adaptation for dialogue format.

### 2. Rich Human Feedback for Multi-Hop Reasoning (Joshi et al., 2023)
- GitHub: `joshinh/rich-feedback-reasoning`
- License: MIT
- 2,361 examples (StrategyQA: 1,565 + Sports Understanding: 796)
- Format: question → model reasoning → correction + error type + error description → corrected answer
- Error types: Factual Error, Missing Facts, Irrelevant Facts, Logical Inconsistency
- PERFECT format match for our pipeline
- Very small but high quality

### 3. FIRE (PengxiangLi, CVPR 2024)
- HuggingFace: `PengxiangLi/FIRE`
- License: CC-BY-4.0
- 1.1M multi-turn conversations (100K GPT-4V + 1M model-generated)
- Format: question → student response → teacher score + feedback → student revision → ...
- Multi-turn feedback-refinement loops
- 27 source datasets
- Caveat: VLM-focused (multimodal/images). Text-only subset may be extractable.

### 4. WildChat + WildReward (AI2/Tsinghua, 2024-2026)
- HuggingFace: `allenai/WildChat` (652K), `allenai/WildChat-1M`, `yuntian-deng/WildChat-4.8M-Full`
- License: ODC-BY 1.0
- ~26.5% of turns contain implicit correction signals (user re-attempts with feedback)
- WildReward (Tsinghua 2026): Extracts user satisfaction levels, 186K WildFB instances
- RLHI (Meta 2025): User-guided rewrites → DPO preference pairs
- Caveat: Requires significant processing to extract explicit correction pairs. Noisy labels.

### 5. LEMMA (2025)
- HuggingFace: `panzs19/LEMMA`
- ~90K synthesized math problems with systematic error injection
- Two correction strategies: Fix & Continue + Fresh & Restart
- Error categories systematically annotated
- Caveat: Math-specific, synthetic (GPT-4o generated)

## Code Correction Datasets

### 6. Vezora/Open-Critic-GPT
- HuggingFace: `Vezora/Open-Critic-GPT` + `Vezora/Code-Preference-Pairs`
- ~55K each: bugged code → explanation → fixed code
- 127 programming languages
- DPO-ready preference pairs

### 7. Coffee-Gym/Coffee (EMNLP 2024)
- HuggingFace: `Coffee-Gym/Project-Coffee-Gym`
- Human code edit traces + written natural language feedback
- Unit-test-based reward functions

## Benchmarks (NOT training data)

### PinchBench (pinchbench/skill)
- GitHub: `github.com/pinchbench/skill`, MIT License
- 23 standardized real-world coding agent tasks
- Leaderboard: pinchbench.com
- Part of OpenClaw ecosystem
- USE: Evaluation benchmark for trained models, not training data
- Each task has: prompt, expected behavior, scoring checklist, automated checks, LLM judge rules

## Chinese Language Datasets

### TW-NLP ChineseErrorCorrectData
- HuggingFace: `twnlp/ChineseErrorCorrectData`
- ~2M dialogue-format error correction pairs
- Covers spelling and grammar errors

### EXCGEC
- 8,216 samples with edit-level explanations
- Error type labels + severity scores + natural language descriptions
- Educational focus

## Preference Datasets (less useful, no correction fields)

### UltraFeedback Binarized
- HuggingFace: `HuggingFaceH4/ultrafeedback_binarized`
- 64K prompts × 4 responses, GPT-4 scored → chosen/rejected pairs
- MIT license
- No explicit correction or feedback text

### Anthropic hh-rlhf
- HuggingFace: `Anthropic/hh-rlhf`
- Chosen/rejected pairs for helpfulness + harmlessness
- No correction fields; known annotation errors

## NOT Publicly Available

### CnR (Critique and Revise) - Amazon, 2023
- Paper only: arxiv 2311.14543
- 1,000 annotated (prompt, initial, critique, revision) pairs
- PERFECT format but dataset NOT released
