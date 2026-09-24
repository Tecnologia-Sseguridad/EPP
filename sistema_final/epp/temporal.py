"""Temporal decisions based on fresh observations, independent of rendering FPS."""
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Evidence:
    samples: deque = field(default_factory=lambda: deque(maxlen=40))
    state: str = "verificando"
    confirmed_at: float = float("-inf")
    window: float = 2.0
    freshness: float = 2.5

    def observe(self, value, now):
        # None = conflicting observations; zero = no conclusion.
        if self.samples and now <= self.samples[-1][0]:
            return
        self.samples.append((now, value))
        while self.samples and now - self.samples[0][0] > self.window:
            self.samples.popleft()
        recent = [(t, v) for t, v in self.samples if v in (-1, 1)]
        if value not in (-1, 1):
            return
        matching = [(t, v) for t, v in recent if v == value]
        minimum = 3 if value == 1 else 4
        # Confirmation needs independent frames AND elapsed observation time.
        if (len(matching) >= minimum and matching[-1][0] - matching[0][0] >= 0.30
                and len(matching) / max(1, len(self.samples)) >= 0.70):
            self.state = "si" if value == 1 else "no"
            self.confirmed_at = now

    def decision(self, now):
        if now - self.confirmed_at > self.freshness:
            return "verificando"
        return self.state


def overall_status(decisions, present=True):
    if not present:
        return "VERIFICANDO"
    if "no" in decisions.values():
        return "INCOMPLETO"
    if decisions and all(value == "si" for value in decisions.values()):
        return "COMPLETO"
    return "VERIFICANDO"
