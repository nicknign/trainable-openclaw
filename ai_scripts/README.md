# ai_scripts — Agent-Generated Temporary Scripts

此目录存放 AI agent (Claude Code / subagent) 在探索、调试、验证过程中生成的一次性脚本。

**与 `scripts/` 的区别：**
- `scripts/` — 项目正式脚本，经过审查，长期维护
- `ai_scripts/` — agent 临时产物，可能粗糙，可能一次性

**Agent 写入规范：**
- 所有 agent 生成的临时/调试脚本必须写入 `ai_scripts/`
- 禁止在 `scripts/`、`trainable_openclaw/` 或其他源码目录生成临时文件
- 文件名使用 `ai_` 前缀或其他清晰标记

**清理规则：**
- 此目录内容可安全清理（`git clean -f ai_scripts/`）
- 如果有临时脚本被证明有长期价值，手动迁移到 `scripts/`
