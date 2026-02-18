from dataclasses import dataclass, field

@dataclass
class Trajectory:
    initial_state: dict = field(default_factory=dict)
    steps: list[dict] = field(default_factory=list)