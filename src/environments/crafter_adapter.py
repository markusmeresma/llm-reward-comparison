import gym as old_gym
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from environments.adapter import EnvAdapter
from models import Trajectory

class CrafterAdapter(EnvAdapter):
    def make_base_env(self, env_id: str, seed: int = None, render_mode=None) -> old_gym.Env:
        gym_env = old_gym.make(env_id)
        env = GymV21CompatibilityV0(env=gym_env)
        return env
    
    # NOTE TO SELF - may need to extract info for crafter somehow else then
    def extract_initial_state(self, env):
        # Crafter doesn't provide info on reset; return empty state
        return {}
    
    def extract_step_state(self, env: old_gym.Env, action: int, info: dict) -> dict:
        return {
            "action": action,
            "pos": tuple(info.get("player_pos", ())),
            "inventory": dict(info.get("inventory", {})),
            "achievements": dict(info.get("achievements", {})),
        }
        
    def trajectory_to_text(self, trajectory: Trajectory, env_id: str, terminated: bool, truncated: bool) -> str:
        steps = trajectory.steps
        lines = [
            f"Environment: {env_id}",
            f"Steps: {len(steps)}",
            f"Outcome: terminated={terminated}, truncated={truncated}",
        ]
        
        # Summarise final inventory state
        if steps:
            last = steps[-1]
            lines.append(f"Final inventory: {last['inventory']}")
            unlocked = [k for k, v in last["achievements"].items() if v > 0]
            lines.append(f"Achievements unlocked: {unlocked if unlocked else 'none'}")
            
        # List actions taken
        action_names = self.action_names
        lines.append("Actions:")
        for i, step in enumerate(steps):
            name = action_names.get(step["action"], f"unknown({step['action']})")
            lines.append(f" {i}: {name} -> pos={step['pos']}")
        return "\n".join(lines)
    
    def is_success(self, reward: float, info: dict):
        # TODO(IMPORTANT): Define explicit Crafter success metrics before implementation
        pass
            
    @property
    def action_names(self) -> dict[int, str]:
        return {
            0: "noop", 1: "move_left", 2: "move_right", 3: "move_up",
            4: "move_down", 5: "do", 6: "sleep", 7: "place_stone",
            8: "place_table", 9: "place_furnace", 10: "place_plant",
            11: "make_wood_pickaxe", 12: "make_stone_pickaxe",
            13: "make_iron_pickaxe", 14: "make_wood_sword",
            15: "make_stone_sword", 16: "make_iron_sword",
        }