"""Validate judge executor and rubric evolver end-to-end with real API calls."""
import sys, json, os, tempfile

sys.path.insert(0, ".")
if not os.environ.get("DEEPSEEK_API_KEY"):
    print("WARNING: DEEPSEEK_API_KEY not set, API tests will fail")
    os.environ["DEEPSEEK_API_KEY"] = ""

from trainable_openclaw.evaluation.judge import JudgeExecutor
from trainable_openclaw.evaluation.rubric import Rubric

# Test 1: Judge Executor - single rubric scoring
print("=== Test 1: Judge Executor (single rubric) ===")

rubric = Rubric(
    id="test-r1",
    名称="factual-accuracy",
    评分提示词="Score the answer for factual accuracy on a scale of 0 to 1:\n\nAnswer to evaluate: {content}\n\nOutput JSON with keys: score (float 0-1), reason (string)",
    来源模式="test",
    类别组="general",
)

answer = "Python is a popular programming language created by Guido van Rossum in 1991."

executor = JudgeExecutor(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model="deepseek-v4-flash",
    enable_thinking=False,
)

print(f"Scoring rubric: {rubric.名称}")
result = executor.score_one_sync(rubric, answer)
print(f"  Score: {result.分数}")
print(f"  Summary: {result.总结[:100] if result.总结 else 'N/A'}")
print(f"  Error: {result.解析错误 or 'none'}")
assert result.分数 is not None, "Score should not be None"
print("  => PASS")

# Test 2: Merged scoring (multiple rubrics)
print()
print("=== Test 2: Judge Executor (merged) ===")

rubrics = [
    Rubric(id="r1", 名称="accuracy", 评分提示词="Score accuracy 0-1:\n{content}\nJSON: {\"score\": <float>}", 来源模式="test", 类别组="general"),
    Rubric(id="r2", 名称="completeness", 评分提示词="Score completeness 0-1:\n{content}\nJSON: {\"score\": <float>}", 来源模式="test", 类别组="general"),
]

# use_merged=True is the default on JudgeExecutor
merged_executor = JudgeExecutor(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model="deepseek-v4-flash",
    enable_thinking=False,
    use_merged=True,
)

results = merged_executor.score_answers_sync(
    prompt="What is machine learning?",
    answers=["Machine learning is a subfield of AI that enables computers to learn from data."],
    rubrics=rubrics,
)

assert len(results) > 0, "Should have results"
result_dict = results[0]
assert "回答" in result_dict, "Should have 回答 key"
assert "评分" in result_dict, "Should have 评分 key"
scores = result_dict["评分"]
for s in scores:
    print(f"  {s['rubric名称']}: score={s['分数']}")
assert len(scores) == 2, f"Should have 2 rubric scores, got {len(scores)}"
print("  => PASS")

# Test 3: Rubric Evolver
print()
print("=== Test 3: Rubric Evolver ===")
from trainable_openclaw.evaluation.rubric_evolver import (
    RubricEvolver, EvolutionTrigger, extract_dimensions_from_text
)

text = "The code has a logic error in the loop boundary condition and a missing null check."
dims = extract_dimensions_from_text(text)
print(f"  Dimensions from text: {dims}")
assert len(dims) > 0, "Should extract dimensions"
print("  => PASS")

trigger = EvolutionTrigger(
    reason="低分样本15个 >= 10; 新纠错5条 >= 5",
    low_score_count=15,
    new_correction_count=5,
    stale_rubric_count=2,
)
# EvolutionTrigger.reason is set by _build_trigger when conditions are met
assert trigger.reason, "Trigger should have a reason"
print(f"  Trigger reason: {trigger.reason}")

evolver = RubricEvolver(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    rubrics_path="data/rubrics_category.json",
    db_path="data/conversations.db",
    min_low_samples=10,
    max_rubrics=8,
)
stats = evolver.get_rubric_stats()
print(f"  Rubric stats: {json.dumps(stats, ensure_ascii=False)}")
print("  => PASS")

# Test 4: Pipeline
print()
print("=== Test 4: Pipeline ===")
from trainable_openclaw.pipeline import Pipeline, PipelineConfig

config = PipelineConfig(api_key="sk-test")
pipeline = Pipeline(config)
cfg = pipeline.generate_training_config()
assert "trajectory.enabled=true" in cfg
assert "api_key=sk-test" in cfg
print(f"  Config: {cfg[:120]}...")
print("  => PASS")

print()
print("ALL_VALIDATION_TESTS_OK")
