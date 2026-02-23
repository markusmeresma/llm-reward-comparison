import gym as old_gym
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from environments.adapter import EnvAdapter
from segment import SegmentResult
from collections import Counter
import crafter #required for env registration

CRAFTER_ACHIEVEMENTS: tuple[str, ...] = (
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
)

# Inventory categories for segment text formatting.
# Crafter's info["inventory"] contains 16 items across these three groups,
# all with max value 9 (see crafter/env.py).
VITAL_KEYS = ("health", "food", "drink", "energy")
MATERIAL_KEYS = ("wood", "stone", "coal", "iron", "diamond", "sapling")
TOOL_KEYS = (
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
)

# Maps craft action names to the inventory key that increases on success.
CRAFT_ACTION_TO_INV_KEY = {
    "make_wood_pickaxe": "wood_pickaxe",
    "make_stone_pickaxe": "stone_pickaxe",
    "make_iron_pickaxe": "iron_pickaxe",
    "make_wood_sword": "wood_sword",
    "make_stone_sword": "stone_sword",
    "make_iron_sword": "iron_sword",
}

# Maps place action names to the inventory key that decreases on success
# (material consumed by placement).
PLACE_ACTION_TO_INV_KEY = {
    "place_table": "wood",
    "place_furnace": "stone",
    "place_plant": "sapling",
    "place_stone": "stone",
}

class CrafterAdapter(EnvAdapter):
    """Adapter for the Crafter environment.
    
    Crafter uses the old OpenAI Gym API (gym<=0.21), so it is wrapped
    with shimmy's GymV21CompatibilityV0 for Gymnasium compatibility.
    """
    def make_base_env(self, env_id: str, seed: int = None, render_mode=None) -> old_gym.Env:
        gym_env = old_gym.make(env_id)
        env = GymV21CompatibilityV0(env=gym_env)
        return env
    
    # NOTE TO SELF - may need to extract info for crafter somehow else then
    def extract_initial_state(self, env):
        # Crafter doesn't provide info on reset; return empty state
        return {}
    
    def extract_step_state(self, env: old_gym.Env, action: int, info: dict) -> dict:
        """Extract per-step state from Crafter's info dict.
        Stores action, position, inventory (16 items), and achievements (22)."""
        return {
            "action": action,
            "pos": tuple(info.get("player_pos", ())),
            "inventory": dict(info.get("inventory", {})),
            "achievements": dict(info.get("achievements", {})),
        }
        
    def _classify_actions(self, steps: list[dict]) -> tuple[Counter, Counter, int]:
        """Classify each step's action as basic, successful craft/place, or failed.

        Compares consecutive inventory snapshots to detect whether craft/place
        actions actually succeeded. Craft/place actions at step 0 are counted
        as failed (no previous inventory to compare against).

        Returns (basic_counts, success_counts, n_failed).
        """
        basic_counts = Counter()
        success_counts = Counter()
        n_failed = 0
        
        for t, step in enumerate(steps):
            name = self.action_names.get(step["action"], f"unknown({step['action']})")
            
            is_craft_or_place = name in CRAFT_ACTION_TO_INV_KEY or name in PLACE_ACTION_TO_INV_KEY
            
            if not is_craft_or_place:
                basic_counts[name] += 1
                continue
            
            # At t==0 there is no previous inventory to compare against,
            # so we can't verify whether the craft/place succeeded.
            # Count it as failed to avoid leaking unverified action names
            # into the Actions line, where the LLM may misinterpret them
            # as successful crafts.
            if t == 0:
                n_failed += 1
                continue
            
            prev_inv = steps[t - 1]["inventory"]
            curr_inv = step["inventory"]
            
            if name in CRAFT_ACTION_TO_INV_KEY:
                key = CRAFT_ACTION_TO_INV_KEY[name]
                succeeded = curr_inv.get(key, 0) > prev_inv.get(key, 0)
            else:
                key = PLACE_ACTION_TO_INV_KEY[name]
                succeeded = curr_inv.get(key, 0) < prev_inv.get(key, 0)
                
            if succeeded:
                success_counts[name] += 1
            else:
                n_failed += 1
                
        return basic_counts, success_counts, n_failed
        
    def segment_to_text(self, result: SegmentResult) -> str:
        """Convert a Crafter segment into a compact text summary for LLM evaluation.
        
        Reports: vital deltas and end values, material/tool deltas (non-zero only),
        newly unlocked achievements, aggregated action counts (movement actions
        collapsed into one group), and number of unique positions visited.
        """
        steps = result.steps
        first_inv = steps[0]["inventory"]
        last_inv = steps[-1]["inventory"]
        
        # Episode status
        if result.episode_ended:
            status = f"episode ended ({result.termination_reason})"
        else:
            status = "episode ongoing"
            
        lines = [f"Segment: {len(steps)} steps ({status})"]
        
        # Vitals: current value and delta
        vital_parts = []
        for k in VITAL_KEYS:
            start = first_inv.get(k, 0)
            end = last_inv.get(k, 0)
            delta = end - start
            sign = "+" if delta >= 0 else ""
            vital_parts.append(f"{k} {start}→{end} ({sign}{delta})")
        lines.append(f"Vitals: {', '.join(vital_parts)}")
        
        # Materials: only show non-zero deltas
        mat_parts = []
        for k in MATERIAL_KEYS:
            delta = last_inv.get(k, 0) - first_inv.get(k, 0)
            if delta != 0:
                sign = "+" if delta > 0 else ""
                mat_parts.append(f"{sign}{delta} {k}")
        if mat_parts:
            lines.append(f"Materials: {', '.join(mat_parts)}")
            
        # Tools: only show non-zero deltas
        tool_parts = []
        for k in TOOL_KEYS:
            delta = last_inv.get(k, 0) - first_inv.get(k, 0)
            if delta != 0:
                sign = "+" if delta > 0 else ""
                tool_parts.append(f"{sign}{delta} {k}")
        if tool_parts:
            lines.append(f"Tools: {', '.join(tool_parts)}")
            
        # Achievements newly unlocked during this segment (went from 0 to >0)
        first_ach = steps[0]["achievements"]
        last_ach = steps[-1]["achievements"]
        new_achievements = [
            name for name in CRAFTER_ACHIEVEMENTS
            if first_ach.get(name, 0) == 0 and last_ach.get(name, 0) > 0
        ]
        if new_achievements:
            lines.append(f"Achievements unlocked: {', '.join(new_achievements)}")
            
        basic_counts, success_counts, n_failed = self._classify_actions(steps)
        
        # Collapse directional movement into one group
        movement_count = sum(
            basic_counts.pop(name, 0)
            for name in ("move_left", "move_right", "move_up", "move_down")
        )
        if movement_count:
            basic_counts["movement"] = movement_count
        basic_str = ", ".join(
            f"{count} {name}" for name, count in basic_counts.most_common()
        )
        lines.append(f"Actions: {basic_str}")
        
        if success_counts:
            success_str = ", ".join(
                f"{count} {name}" for name, count in success_counts.most_common()
            )
            lines.append(f"Successful crafts/placements: {success_str}")

        if n_failed > 0:
            lines.append(f"Failed craft/place attempts: {n_failed}")
        
        # Exploration
        unique_positions = len(set(s["pos"] for s in steps))
        lines.append(f"Exploration: {unique_positions} unique positions")

        return "\n".join(lines)
            
    @property
    def action_names(self) -> dict[int, str]:
        """Crafter's 17 discrete actions (from crafter/data.yaml)."""
        return {
            0: "noop", 1: "move_left", 2: "move_right", 3: "move_up",
            4: "move_down", 5: "do", 6: "sleep", 7: "place_stone",
            8: "place_table", 9: "place_furnace", 10: "place_plant",
            11: "make_wood_pickaxe", 12: "make_stone_pickaxe",
            13: "make_iron_pickaxe", 14: "make_wood_sword",
            15: "make_stone_sword", 16: "make_iron_sword",
        }
        
    @property
    def achievement_names(self) -> tuple[str, ...]:
        return CRAFTER_ACHIEVEMENTS
    
    def achievements_binary(self, achievements: dict) -> list[int]:
        """Return fixed-order 22-dim binary vector from Crafter achievements dict."""
        return [1 if achievements.get(name, 0) >= 1 else 0 for name in self.achievement_names]
    
    def achievement_column_names(self) -> list[str]:
        """CSV column names for per-achievement binary flags."""
        return [f"achievement_{name}" for name in self.achievement_names]
    
    def success_rate_column_names(self) -> list[str]:
        """CSV column names for per-achievement success rates."""
        return [f"success_rate_{name}" for name in self.achievement_names]