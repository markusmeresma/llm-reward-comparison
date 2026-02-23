import gymnasium as gym
from collections import Counter
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from minigrid.core.grid import Grid
from environments.adapter import EnvAdapter
from segment import SegmentResult

class MiniGridAdapter(EnvAdapter):
    """Adapter for MiniGrid environments.
    
    Caches the goal position on reset so segment_to_text() can reference it.
    """
    ACTION_NAMES = {
        0: "turn_left", 1: "turn_right", 2: "forward",
        3: "pickup", 4: "drop", 5: "toggle", 6: "done",
    }
    
    def __init__(self):
        self._goal_pos = None
    
    def make_base_env(self, env_id, seed=None, render_mode=None):
        env = gym.make(env_id, render_mode=render_mode)
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        return env
    
    def extract_initial_state(self, env):
        """Extract agent start position/direction and goal position on reset."""
        unwrapped = env.unwrapped
        self._goal_pos = self._get_goal_pos(unwrapped.grid)
        return {
            "pos": unwrapped.agent_pos,
            "dir": unwrapped.agent_dir,
            "goal_pos": self._goal_pos,
        }
        
    def extract_step_state(self, env, action, info):
        unwrapped = env.unwrapped
        return {
            "action": action,
            "pos": unwrapped.agent_pos,
            "dir": unwrapped.agent_dir,
        }
        
    def segment_to_text(self, result: SegmentResult) -> str:
        """Convert a MiniGrid segment into text for LLM evaluation.
        
        Reports: start/end positions and directions, goal position,
        whether the goal was reached, action counts, and unique positions visited.
        """
        steps = result.steps
        first = steps[0]
        last = steps[-1]
        
        if result.episode_ended:
            status = "episode ended"
        else:
            status = "episode ongoing"
            
        reached_goal = (
            result.episode_ended
            and self._goal_pos is not None
            and tuple(last["pos"]) == tuple(self._goal_pos)
        )
        
        lines = [f"Segment: {len(steps)} steps ({status})"]
        lines.append(f"Start: pos={tuple(first['pos'])}, dir={first['dir']}, goal={self._goal_pos}")
        lines.append(f"End: pos={tuple(last['pos'])}, dir={last['dir']}")
        
        if reached_goal:
            lines.append("Reached goal: yes")
        
        action_counts = Counter(
            self.ACTION_NAMES.get(s["action"], f"unknown({s['action']})")
            for s in steps
        )
        action_str = ", ".join(
            f"{count} {name}" for name, count in action_counts.most_common()
        )
        lines.append(f"Actions: {action_str}")
        
        unique_positions = len(set(tuple(s["pos"]) for s in steps))
        lines.append(f"Unique positions: {unique_positions}")

        return "\n".join(lines)
    
    def is_success(self, reward, info):
        """MiniGrid success = any positive reward (reached goal)."""
        return reward > 0
    
    @property
    def action_names(self):
        return self.ACTION_NAMES
    
    def _get_goal_pos(self, grid: Grid):
        """Find the goal cell position by scanning the grid."""
        for x in range(grid.width):
            for y in range(grid.height):
                obj = grid.get(x, y)
                if obj is not None and obj.type == "goal":
                    return (x, y)
        return None
    
    