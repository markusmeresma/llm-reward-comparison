import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from minigrid.core.grid import Grid
from environments.adapter import EnvAdapter

class MiniGridAdapter(EnvAdapter):
    ACTION_NAMES = {
        0: "turn_left", 1: "turn_right", 2: "forward",
        3: "pickup", 4: "drop", 5: "toggle", 6: "done",
    }
    
    def make_base_env(self, env_id, seed=None, render_mode=None):
        env = gym.make(env_id, render_mode=render_mode)
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        return env
    
    def extract_initial_state(self, env):
        unwrapped = env.unwrapped
        return {
            "pos": unwrapped.agent_pos,
            "dir": unwrapped.agent_dir,
            "goal_pos": self._get_goal_pos(unwrapped.grid),
        }
        
    def extract_step_state(self, env, action, info):
        unwrapped = env.unwrapped
        return {
            "action": action,
            "pos": unwrapped.agent_pos,
            "dir": unwrapped.agent_dir,
        }
        
    def trajectory_to_text(self, trajectory, env_id, terminated, truncated):
        init = trajectory.initial_state
        steps = trajectory.steps
        lines = [
            f"Environment: {env_id}",
            f"Start: pos={init['pos']}, dir={init['dir']}",
            f"Goal: pos={init['goal_pos']}",
            f"Steps: {len(steps)}",
            f"Outcome: terminated={terminated}, truncated={truncated}",
            "Actions:",
        ]
        for i, step in enumerate(steps):
            name = self.ACTION_NAMES.get(step["action"], f"unknown({step['action']})")
            lines.append(f" {i}: {name} -> pos={step['pos']}")
        return "\n".join(lines)
    
    def is_success(self, reward, info):
        return reward > 0
    
    @property
    def action_names(self):
        return self.ACTION_NAMES
    
    def _get_goal_pos(self, grid: Grid):
        for x in range(grid.width):
            for y in range(grid.height):
                obj = grid.get(x, y)
                if obj is not None and obj.type == "goal":
                    return (x, y)
        return None
    
    