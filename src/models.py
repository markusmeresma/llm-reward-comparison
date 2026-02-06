from dataclasses import dataclass
from minigrid.core.grid import Grid

@dataclass
class Step:
    action: int
    pos: tuple
    dir: int

@dataclass
class Trajectory:
    initial_pos: tuple
    initial_dir: int
    goal_pos: tuple
    steps: list[Step]
    outcome: dict = None
    
def get_goal_pos(grid: Grid) -> tuple:
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj is not None and obj.type == "goal":
                goal_pos = (x, y)
                return goal_pos