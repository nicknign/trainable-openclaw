"""Smoke tests for TrainingOrchestrator — no GPU required.

Verify idle detection, sample queue, training trigger conditions,
mode switching, and 503 behavior during training.

Run with:
    python -m pytest tests/test_orchestrator.py -v
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from trainable_openclaw.training.orchestrator import TrainingOrchestrator, TrainingSample


# ---------------------------------------------------------------------------
# TrainingSample
# ---------------------------------------------------------------------------


class TestTrainingSample:
    def test_defaults(self):
        before = time.time()
        s = TrainingSample(prompt_ids=[1, 2, 3], response_ids=[10, 20])
        after = time.time()
        assert s.prompt_ids == [1, 2, 3]
        assert s.response_ids == [10, 20]
        assert s.metadata == {}
        assert before <= s.timestamp <= after

    def test_with_metadata(self):
        s = TrainingSample(
            prompt_ids=[1],
            response_ids=[2],
            metadata={"source": "test"},
        )
        assert s.metadata == {"source": "test"}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_values(self):
        orch = TrainingOrchestrator()
        assert orch.idle_timeout == 60.0
        assert orch.min_samples == 16
        assert orch.sample_count == 0
        assert orch.mode == "serving"
        assert orch.training_in_progress is False

    def test_custom_values(self):
        orch = TrainingOrchestrator(idle_timeout=10.0, min_samples=8, max_buffer_size=100)
        assert orch.idle_timeout == 10.0
        assert orch.min_samples == 8
        assert orch.max_buffer_size == 100

    def test_rejects_negative_idle_timeout(self):
        with pytest.raises(ValueError):
            TrainingOrchestrator(idle_timeout=0)

    def test_rejects_negative_min_samples(self):
        with pytest.raises(ValueError):
            TrainingOrchestrator(min_samples=0)


# ---------------------------------------------------------------------------
# Request recording
# ---------------------------------------------------------------------------


class TestRecordRequest:
    def test_adds_sample_and_bumps_timestamp(self):
        orch = TrainingOrchestrator()
        before = time.time()
        orch.record_request([1, 2], [3, 4])
        assert orch.sample_count == 1
        assert orch.idle_seconds < (time.time() - before) + 0.1

    def test_accumulates_multiple(self):
        orch = TrainingOrchestrator()
        for i in range(10):
            orch.record_request([i], [i * 10])
        assert orch.sample_count == 10

    def test_enforces_max_buffer_size(self):
        orch = TrainingOrchestrator(max_buffer_size=3)
        for i in range(5):
            orch.record_request([i], [i])
        assert orch.sample_count == 3


# ---------------------------------------------------------------------------
# Training trigger condition
# ---------------------------------------------------------------------------


class TestShouldTrain:
    def test_returns_false_when_busy(self):
        orch = TrainingOrchestrator(min_samples=1)
        # Just recorded a request → not idle
        orch.record_request([1], [2])
        assert orch.should_train() is False

    def test_returns_false_when_not_enough_samples(self):
        orch = TrainingOrchestrator(idle_timeout=1e-9, min_samples=5)
        orch.record_request([1], [2])
        time.sleep(0.01)
        assert orch.should_train() is False

    def test_returns_true_when_idle_and_enough(self):
        orch = TrainingOrchestrator(idle_timeout=1e-9, min_samples=2)
        orch.record_request([1], [2])
        orch.record_request([3], [4])
        time.sleep(0.01)
        assert orch.should_train() is True

    def test_returns_false_during_training(self):
        orch = TrainingOrchestrator(idle_timeout=1e-9, min_samples=1)
        orch.record_request([1], [2])
        time.sleep(0.01)
        orch._training_in_progress = True
        assert orch.should_train() is False


# ---------------------------------------------------------------------------
# Training callback execution
# ---------------------------------------------------------------------------


class TestTriggerTraining:
    def test_invokes_callback_with_samples(self):
        orch = TrainingOrchestrator()
        captured = []

        def cb(samples):
            captured.extend(samples)

        orch.set_train_fn(cb)
        orch.record_request([1], [2])
        orch.record_request([3], [4])

        orch._trigger_training()

        assert len(captured) == 2
        assert captured[0].prompt_ids == [1]
        assert captured[1].prompt_ids == [3]

    def test_clears_buffer_after_training(self):
        orch = TrainingOrchestrator()
        orch.set_train_fn(lambda s: None)
        orch.record_request([1], [2])
        orch.record_request([3], [4])

        orch._trigger_training()

        assert orch.sample_count == 0

    def test_mode_switches_to_serving_after_training(self):
        orch = TrainingOrchestrator()
        orch.set_train_fn(lambda s: None)
        orch.record_request([1], [2])

        orch._trigger_training()

        assert orch.mode == "serving"
        assert orch.training_in_progress is False

    def test_mode_switches_to_serving_on_error(self):
        orch = TrainingOrchestrator()

        def failing_cb(samples):
            raise RuntimeError("train failed")

        orch.set_train_fn(failing_cb)
        orch.record_request([1], [2])

        orch._trigger_training()  # should not raise

        assert orch.mode == "serving"
        assert orch.training_in_progress is False

    def test_warns_when_no_callback(self):
        orch = TrainingOrchestrator()
        orch.record_request([1], [2])
        # Should not crash even without a registered callback
        orch._trigger_training()
        assert orch.mode == "serving"


# ---------------------------------------------------------------------------
# Monitoring loop
# ---------------------------------------------------------------------------


class TestMonitorLoop:
    def test_triggers_when_conditions_met(self):
        orch = TrainingOrchestrator(idle_timeout=1e-9, min_samples=2)
        captured = []
        orch.set_train_fn(lambda s: captured.extend(s))

        orch.record_request([1], [2])
        orch.record_request([3], [4])

        orch.start_monitoring(poll_interval=0.05)
        time.sleep(0.3)  # Give monitor thread time to detect
        orch.stop_monitoring()

        assert len(captured) >= 2

    def test_does_not_trigger_when_busy(self):
        orch = TrainingOrchestrator(idle_timeout=999.0, min_samples=1)
        triggered = []

        def cb(samples):
            triggered.append(True)

        orch.set_train_fn(cb)
        orch.record_request([1], [2])

        orch.start_monitoring(poll_interval=0.05)
        time.sleep(0.2)
        orch.stop_monitoring()

        assert len(triggered) == 0  # idle_timeout too large

    def test_idempotent_start(self):
        orch = TrainingOrchestrator()
        orch.start_monitoring()
        t1 = orch._monitor_thread
        orch.start_monitoring()  # second call should be no-op
        t2 = orch._monitor_thread
        orch.stop_monitoring()
        assert t1 is t2


# ---------------------------------------------------------------------------
# Thread safety: concurrent record_request
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_records(self):
        orch = TrainingOrchestrator(max_buffer_size=10000)
        errors = []

        def record_many(n):
            try:
                for i in range(n):
                    orch.record_request([i], [i * 10])
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_many, args=(500,))
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # With max_buffer_size=10000, the queue should hold at most 10000
        assert orch.sample_count <= 10000
        assert orch.sample_count > 0


# ---------------------------------------------------------------------------
# Idle tracking
# ---------------------------------------------------------------------------


class TestIdleTracking:
    def test_idle_seconds_increases_over_time(self):
        orch = TrainingOrchestrator()
        orch.record_request([1], [2])
        idle_1 = orch.idle_seconds
        time.sleep(0.15)
        idle_2 = orch.idle_seconds
        assert idle_2 > idle_1

    def test_record_request_resets_idle(self):
        orch = TrainingOrchestrator()
        orch.record_request([1], [2])
        time.sleep(0.1)
        orch.record_request([3], [4])
        # idle should be reset to near-zero
        assert orch.idle_seconds < 0.05
