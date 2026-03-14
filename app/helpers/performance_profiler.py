import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class StageStats:
    total_ms: float = 0.0
    calls: int = 0

    @property
    def avg_ms(self) -> float:
        if self.calls <= 0:
            return 0.0
        return self.total_ms / self.calls


class PipelineProfiler:
    """Thread-safe per-stage profiler for end-to-end video throughput diagnostics."""

    def __init__(self, enabled: bool = False, log_every_n_frames: int = 60):
        self.enabled = enabled
        self.log_every_n_frames = max(1, int(log_every_n_frames))
        self._lock = threading.Lock()
        self._stats: dict[str, StageStats] = defaultdict(StageStats)
        self._start_time = time.perf_counter()
        self._frames = 0

    def reset(self, enabled: bool | None = None):
        with self._lock:
            if enabled is not None:
                self.enabled = enabled
            self._stats = defaultdict(StageStats)
            self._start_time = time.perf_counter()
            self._frames = 0

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.add_timing(name, elapsed_ms)

    def add_timing(self, name: str, elapsed_ms: float):
        if not self.enabled:
            return
        with self._lock:
            st = self._stats[name]
            st.total_ms += float(elapsed_ms)
            st.calls += 1

    def mark_frame_done(self) -> str | None:
        if not self.enabled:
            return None
        with self._lock:
            self._frames += 1
            if self._frames % self.log_every_n_frames != 0:
                return None
            return self._build_report_locked()

    def _build_report_locked(self) -> str:
        elapsed = max(time.perf_counter() - self._start_time, 1e-6)
        fps = self._frames / elapsed
        stage_parts: list[str] = []
        for stage_name in sorted(self._stats.keys()):
            stats = self._stats[stage_name]
            stage_parts.append(
                f"{stage_name}: avg={stats.avg_ms:.2f}ms total={stats.total_ms:.1f}ms calls={stats.calls}"
            )
        stages_text = " | ".join(stage_parts) if stage_parts else "no stage timings yet"
        return (
            f"[PROFILE] frames={self._frames} elapsed={elapsed:.2f}s fps={fps:.2f} "
            f"stages=[{stages_text}]"
        )

