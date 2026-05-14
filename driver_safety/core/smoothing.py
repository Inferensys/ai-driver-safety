from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class RollingWindow:
    size: int
    values: deque[float] = field(default_factory=deque)

    def add(self, value: float) -> float:
        self.values.append(float(value))
        while len(self.values) > self.size:
            self.values.popleft()
        return self.mean

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


class SignalSmoother:
    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        self._windows: dict[str, RollingWindow] = defaultdict(
            lambda: RollingWindow(size=window_size)
        )

    def update(self, signals: dict[str, float]) -> dict[str, float]:
        return {name: self._windows[name].add(value) for name, value in signals.items()}
