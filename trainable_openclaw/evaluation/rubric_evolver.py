from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EvolutionTrigger:
    reason: str = ""
    low_score_count: int = 0
    low_score_threshold: float = 0.3
    min_low_samples: int = 10
    new_correction_count: int = 0
    min_new_corrections: int = 5
    stale_rubric_count: int = 0


@dataclass
class EvolutionResult:
    triggered: bool = False
    trigger_reason: str = ""
    new_rubrics: int = 0
    updated_rubrics: int = 0
    archived_rubrics: int = 0
    total_active_after: int = 0
    elapsed_seconds: float = 0.0
    details: list[str] = field(default_factory=list)


class RubricEvolver:
    """Log-driven rubric auto-evolution engine.

    Monitors conversation store for low-scoring answers and triggers
    rubric generation/update via RubricEngine when thresholds are met.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "deepseek-v4-flash",
        rubrics_path: str = "data/rubrics_category.json",
        db_path: str = "data/conversations.db",
        evolution_interval_hours: float = 24.0,
        min_low_samples: int = 10,
        min_new_corrections: int = 5,
        max_rubrics: int = 8,
        stale_days: int = 30,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model
        self.rubrics_path = rubrics_path
        self.db_path = db_path
        self.evolution_interval_hours = evolution_interval_hours
        self.min_low_samples = min_low_samples
        self.min_new_corrections = min_new_corrections
        self.max_rubrics = max_rubrics
        self.stale_days = stale_days

    # --- Analysis methods ---

    def _get_low_score_answers(self, store, threshold: float = 0.3) -> list[dict]:
        """Query telemetry for low-rated answers."""
        results = []
        try:
            cur = store.conn.execute(
                "SELECT t.session_id, t.rating, t.correction, t.created_at, "
                "m.content as answer, m.prompt "
                "FROM telemetry_events t "
                "LEFT JOIN messages m ON m.session_id = t.session_id AND m.role = 'assistant' "
                "WHERE t.rating IS NOT NULL AND t.rating < ? "
                "ORDER BY t.created_at DESC LIMIT 200",
                (threshold,),
            )
            for row in cur.fetchall():
                results.append({
                    "session_id": row[0],
                    "rating": row[1],
                    "correction": row[2] or "",
                    "created_at": row[3],
                    "answer": row[4] or "",
                    "prompt": row[5] or "",
                })
        except Exception:
            logger.debug("Failed to query low-score answers", exc_info=True)
        return results

    def _get_recent_corrections(self, store, since_hours: float = 24.0) -> list[dict]:
        """Get recent user corrections from telemetry."""
        results = []
        try:
            cur = store.conn.execute(
                "SELECT t.session_id, t.correction, t.rating, t.created_at, "
                "m.content as answer, m.prompt "
                "FROM telemetry_events t "
                "LEFT JOIN messages m ON m.session_id = t.session_id AND m.role = 'assistant' "
                "WHERE t.correction IS NOT NULL AND t.correction != '' "
                f"AND t.created_at > datetime('now', '-{since_hours} hours') "
                "ORDER BY t.created_at DESC LIMIT 100"
            )
            for row in cur.fetchall():
                results.append({
                    "session_id": row[0],
                    "correction": row[1] or "",
                    "rating": row[2],
                    "created_at": row[3],
                    "answer": row[4] or "",
                    "prompt": row[5] or "",
                })
        except Exception:
            logger.debug("Failed to query recent corrections", exc_info=True)
        return results

    def _build_trigger(self, store) -> EvolutionTrigger:
        """Analyze logs and build a trigger assessment."""
        trigger = EvolutionTrigger()
        trigger.low_score_threshold = 0.3
        trigger.min_low_samples = self.min_low_samples
        trigger.min_new_corrections = self.min_new_corrections

        low_scores = self._get_low_score_answers(store)
        trigger.low_score_count = len(low_scores)

        corrections = self._get_recent_corrections(store, self.evolution_interval_hours)
        trigger.new_correction_count = len(corrections)

        # Check for stale rubrics
        from trainable_openclaw.evaluation.rubric import RubricStore
        try:
            rs = RubricStore(self.rubrics_path)
            stale = [r for r in rs.list_active() if getattr(r, "命中次数", 0) == 0]
            trigger.stale_rubric_count = len(stale)
        except Exception:
            pass

        reasons = []
        if trigger.low_score_count >= trigger.min_low_samples:
            reasons.append(f"低分样本{trigger.low_score_count}个 >= {trigger.min_low_samples}")
        if trigger.new_correction_count >= trigger.min_new_corrections:
            reasons.append(f"新纠错{trigger.new_correction_count}条 >= {trigger.min_new_corrections}")
        if trigger.stale_rubric_count > 0:
            reasons.append(f"{trigger.stale_rubric_count}条过期rubric需要归档")

        trigger.reason = "; ".join(reasons) if reasons else ""
        return trigger

    def _extract_trajectory_like(self, corrections: list[dict]) -> list[dict]:
        """Convert telemetry corrections to trajectory-compatible format."""
        trajectories = []
        for c in corrections:
            traj = {
                "种子提示词": c.get("prompt", ""),
                "类别": c.get("category", ""),
                "最终判定": "纠正后通过" if c.get("rating", 0) and c["rating"] < 3 else "直接通过",
                "纠错次数": 1 if c.get("correction") else 0,
                "最终回答": c.get("answer", ""),
                "对话消息": [],
            }
            if c.get("prompt"):
                traj["对话消息"].append({"role": "user", "content": c["prompt"]})
            if c.get("answer"):
                traj["对话消息"].append({"role": "assistant", "content": c["answer"]})
            if c.get("correction"):
                dims = extract_dimensions_from_text(c["correction"])
                traj["对话消息"].append({
                    "role": "user",
                    "content": c["correction"],
                    "维度": dims,
                })
            trajectories.append(traj)
        return trajectories

    # --- Evolution methods ---

    async def check_and_evolve(self) -> EvolutionResult:
        """Check if evolution should trigger, and run it if so."""
        start = time.time()
        result = EvolutionResult()

        try:
            from trainable_openclaw.logging.conversation_store import ConversationStore
        except Exception:
            result.details.append("无法导入ConversationStore")
            return result

        store = ConversationStore(self.db_path)
        try:
            trigger = self._build_trigger(store)
            result.trigger_reason = trigger.reason

            if not trigger.reason:
                result.details.append("触发条件未满足，跳过演进")
                return result

            result.triggered = True
            result.details.append(f"触发原因: {trigger.reason}")

            # Archive stale rubrics
            if trigger.stale_rubric_count > 0:
                archived = self.archive_stale(store)
                result.archived_rubrics = archived
                result.details.append(f"归档{archived}条过期rubric")

            # Run rubric evolution via RubricEngine
            try:
                from trainable_openclaw.evaluation.rubric_engine import RubricEngine

                corrections = self._get_recent_corrections(
                    store, self.evolution_interval_hours
                )
                trajectories = self._extract_trajectory_like(corrections)
                if not trajectories:
                    result.details.append("无可用于演进的纠错数据")
                    return result

                # Write to temp file for RubricEngine
                tmp_path = None
                try:
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", prefix="evolve_")
                    os.close(tmp_fd)
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        for t in trajectories:
                            f.write(json.dumps(t, ensure_ascii=False) + "\n")

                    engine = RubricEngine(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        model=self.model,
                        max_rubrics=self.max_rubrics,
                    )

                    new_rubrics = await engine.run([tmp_path])
                    if new_rubrics:
                        engine.save(new_rubrics, self.rubrics_path)
                        result.new_rubrics = len(new_rubrics)
                        result.details.append(f"生成{len(new_rubrics)}条新rubric")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                # Get final active count
                from trainable_openclaw.evaluation.rubric import RubricStore
                rs = RubricStore(self.rubrics_path)
                result.total_active_after = len(rs.list_active())

            except Exception as e:
                result.details.append(f"Rubric演进失败: {e}")
                logger.warning("Rubric evolution failed", exc_info=True)

        finally:
            store.close()

        result.elapsed_seconds = time.time() - start
        return result

    async def force_evolve(
        self, trajectory_files: list[str] | None = None
    ) -> EvolutionResult:
        """Force evolution regardless of thresholds."""
        start = time.time()
        result = EvolutionResult(triggered=True, trigger_reason="强制触发")

        try:
            from trainable_openclaw.evaluation.rubric_engine import RubricEngine

            engine = RubricEngine(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                max_rubrics=self.max_rubrics,
            )

            if trajectory_files:
                new_rubrics = await engine.run(trajectory_files)
            else:
                from trainable_openclaw.logging.conversation_store import ConversationStore
                store = ConversationStore(self.db_path)
                try:
                    corrections = self._get_recent_corrections(store, 720.0)
                    trajectories = self._extract_trajectory_like(corrections)
                    if not trajectories:
                        result.details.append("无可用于演进的纠错数据")
                        return result

                    tmp_path = None
                    try:
                        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", prefix="force_evolve_")
                        os.close(tmp_fd)
                        with open(tmp_path, "w", encoding="utf-8") as f:
                            for t in trajectories:
                                f.write(json.dumps(t, ensure_ascii=False) + "\n")
                        new_rubrics = await engine.run([tmp_path])
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                finally:
                    store.close()

            if new_rubrics:
                engine.save(new_rubrics, self.rubrics_path)
                result.new_rubrics = len(new_rubrics)
                result.details.append(f"强制生成{len(new_rubrics)}条新rubric")

            from trainable_openclaw.evaluation.rubric import RubricStore
            rs = RubricStore(self.rubrics_path)
            result.total_active_after = len(rs.list_active())

        except Exception as e:
            result.details.append(f"强制演进失败: {e}")
            logger.warning("Force evolve failed", exc_info=True)

        result.elapsed_seconds = time.time() - start
        return result

    # --- Lifecycle management ---

    def archive_stale(self, store=None, days: int | None = None) -> int:
        """Archive rubrics with zero hits."""
        days = days or self.stale_days
        try:
            from trainable_openclaw.evaluation.rubric import RubricStore
            rs = RubricStore(self.rubrics_path)
            before = len(rs.list_active())
            rs.archive_stale(days)
            after = len(rs.list_active())
            return before - after
        except Exception:
            logger.debug("Failed to archive stale rubrics", exc_info=True)
            return 0

    def get_rubric_stats(self) -> dict:
        """Return current rubric statistics."""
        try:
            from trainable_openclaw.evaluation.rubric import RubricStore
            rs = RubricStore(self.rubrics_path)
            active = rs.list_active()
            all_count = len(rs.rubrics)

            return {
                "活跃数": len(active),
                "总数": all_count,
                "归档数": all_count - len(active),
                "命中分布": {
                    getattr(r, "名称", "?"): getattr(r, "命中次数", 0)
                    for r in active
                },
            }
        except Exception:
            return {"活跃数": 0, "总数": 0, "归档数": 0, "命中分布": {}}


# --- Dimension extraction ---

_KEYWORDS = {
    "代码正确性": ["逻辑错误", "bug", "边界", "异常", "空值", "索引"],
    "命名规范": ["变量名", "命名", "函数名", "PEP", "表意"],
    "类型注解": ["类型", "typing", "注解", "返回类型"],
    "错误处理": ["异常", "try", "catch", "错误处理", "容错"],
    "性能": ["复杂度", "性能", "效率", "O(n)", "优化"],
    "文笔流畅": ["流畅", "通顺", "节奏", "句子"],
    "意境表达": ["意境", "画面感", "用词", "优美", "表达"],
    "结构逻辑": ["结构", "段落", "组织", "论证", "逻辑"],
    "计算准确性": ["计算", "数值", "结果", "错误", "准确"],
    "步骤完整性": ["步骤", "跳步", "省略", "完整", "推导"],
    "表达严谨性": ["严谨", "定义", "定理", "引用", "假设"],
    "信息完整性": ["遗漏", "缺失", "完整", "不全面"],
    "清晰易懂": ["清晰", "理解", "初学者", "易懂"],
    "事实准确性": ["事实", "编造", "错误", "不准确"],
    "安全性": ["安全", "注入", "权限", "SQL", "XSS"],
    "可维护性": ["维护", "SOLID", "可读", "重构"],
}


def extract_dimensions_from_text(text: str) -> list[str]:
    """Extract error dimensions from text using keyword matching."""
    dims = []
    for dim, kws in _KEYWORDS.items():
        if any(kw in text for kw in kws):
            dims.append(dim)
    if not dims:
        dims.append("其他")
    return dims


def extract_dimensions_from_telemetry(store) -> list[dict]:
    """Extract error dimensions from telemetry events.

    Returns list of {维度, 次数, 示例}.
    """
    from collections import defaultdict
    dim_counts: dict[str, int] = defaultdict(int)
    dim_examples: dict[str, list[str]] = defaultdict(list)

    try:
        cur = store.conn.execute(
            "SELECT correction FROM telemetry_events "
            "WHERE correction IS NOT NULL AND correction != '' "
            "ORDER BY created_at DESC LIMIT 500"
        )
        for row in cur.fetchall():
            dims = extract_dimensions_from_text(row[0] or "")
            for d in dims:
                dim_counts[d] += 1
                if len(dim_examples[d]) < 3:
                    dim_examples[d].append(row[0][:120])
    except Exception:
        pass

    return [
        {
            "维度": dim,
            "次数": dim_counts[dim],
            "示例": dim_examples[dim],
        }
        for dim in sorted(dim_counts, key=dim_counts.get, reverse=True)
    ]
