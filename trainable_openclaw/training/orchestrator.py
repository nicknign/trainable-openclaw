"""
Idle detection and training trigger for veRL serve loop.

The TrainingOrchestrator runs alongside the inference server. Requests
are recorded as training samples. When the server goes idle (no requests
for idle_timeout seconds) AND enough samples have accumulated, it triggers
a training cycle via a user-registered callback.

Lifecycle:
    SERVING ──(idle + enough samples)──> TRAINING ──(callback done)──> SERVING

During training, incoming requests receive HTTP 503.
"""

import threading
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Callback signature: (samples: list[TrainingSample]) -> None
TrainCallback = Callable[..., None]


@dataclass
class TrainingSample:
    """A single conversation turn captured for training."""

    prompt_ids: list[int]
    response_ids: list[int]
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class TrainingOrchestrator:
    """Monitors idle state and triggers training when conditions are met.

    Thread-safe: API handlers call record_request() from async context while
    the monitor thread calls should_train() / _trigger_training().
    """

    def __init__(
        self,
        idle_timeout: float = 60.0,
        min_samples: int = 16,
        max_buffer_size: int = 10000,
    ):
        if idle_timeout <= 0:
            raise ValueError("idle_timeout must be positive")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")

        self.idle_timeout = idle_timeout
        self.min_samples = min_samples
        self.max_buffer_size = max_buffer_size

        self._samples: deque[TrainingSample] = deque()
        self._last_request_time: float = time.time()
        self._lock = threading.Lock()
        self._mode: str = "serving"
        self._training_in_progress: bool = False

        self._train_fn: Optional[TrainCallback] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def training_in_progress(self) -> bool:
        return self._training_in_progress

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._samples)

    @property
    def idle_seconds(self) -> float:
        with self._lock:
            return time.time() - self._last_request_time

    # ------------------------------------------------------------------
    # Request recording (called from API handler)
    # ------------------------------------------------------------------

    def record_request(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        metadata: dict | None = None,
    ) -> None:
        """Record a completed generation as a training sample.

        Called by the chat completions endpoint after a successful generation.
        Also bumps the last-request timestamp to reset idle detection.
        """
        with self._lock:
            self._last_request_time = time.time()
            sample = TrainingSample(
                prompt_ids=list(prompt_ids),
                response_ids=list(response_ids),
                metadata=dict(metadata or {}),
            )
            self._samples.append(sample)
            while len(self._samples) > self.max_buffer_size:
                self._samples.popleft()

    # ------------------------------------------------------------------
    # Training condition
    # ------------------------------------------------------------------

    def should_train(self) -> bool:
        """Return True when both conditions hold: idle AND enough samples."""
        if self._training_in_progress:
            return False
        with self._lock:
            idle_long_enough = (time.time() - self._last_request_time) >= self.idle_timeout
            enough_samples = len(self._samples) >= self.min_samples
        return idle_long_enough and enough_samples

    # ------------------------------------------------------------------
    # Training callback registration
    # ------------------------------------------------------------------

    def set_train_fn(self, fn: TrainCallback) -> None:
        """Register the function called when training triggers.

        The callback receives the list of accumulated samples.  It runs
        synchronously in the monitor thread, so the orchestrator will not
        check conditions again until it returns.
        """
        self._train_fn = fn

    # ------------------------------------------------------------------
    # Monitoring lifecycle
    # ------------------------------------------------------------------

    def start_monitoring(self, poll_interval: float = 1.0) -> None:
        """Launch the background monitor thread.

        Args:
            poll_interval: Seconds between idle checks.
        """
        if self._monitor_thread is not None:
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(poll_interval,),
            daemon=True,
            name="train-orchestrator",
        )
        self._monitor_thread.start()
        logger.info(
            "Training orchestrator started — idle_timeout=%ss, min_samples=%d, "
            "poll_interval=%ss",
            self.idle_timeout,
            self.min_samples,
            poll_interval,
        )

    def stop_monitoring(self, timeout: float = 10.0) -> None:
        """Stop the monitor thread and wait for it to exit."""
        self._stop_event.set()
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=timeout)
            self._monitor_thread = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _monitor_loop(self, poll_interval: float) -> None:
        """Background loop: periodically check idle and trigger training."""
        while not self._stop_event.is_set():
            if self.should_train():
                self._trigger_training()
            self._stop_event.wait(timeout=poll_interval)

    def _trigger_training(self) -> None:
        """Drain the sample buffer and invoke the registered training callback."""
        if self._train_fn is None:
            logger.warning("No training function registered — skipping training trigger")
            return

        self._training_in_progress = True
        self._mode = "training"
        t_start = time.time()

        with self._lock:
            batch = list(self._samples)
            self._samples.clear()

        logger.info(
            "Training triggered — %d samples, idle=%.1fs",
            len(batch),
            time.time() - self._last_request_time,
        )

        try:
            self._train_fn(batch)
            elapsed = time.time() - t_start
            logger.info("Training complete in %.1fs, resuming inference", elapsed)
        except Exception:
            logger.exception("Training failed — resuming inference with old weights")
        finally:
            self._mode = "serving"
            self._training_in_progress = False
