CRAFTER_ENV_CONTEXT = """\
Crafter is a 2D open-world survival game. The agent must survive and progress \
through a technology tree by gathering resources, crafting tools, building \
structures, and fighting monsters. Episodes last ~1000 steps.

Technology tree (dependency chain):
  Gathering (no tool needed): wood, sapling, drink
  Gathering (wood_pickaxe): stone, coal
  Gathering (stone_pickaxe): iron
  Gathering (iron_pickaxe): diamond
  Food: eat_cow, eat_plant
  Placement: place_table (wood), place_furnace (stone), place_plant (sapling), place_stone (stone)
  Crafting: wood_pickaxe/sword (wood + table), stone_pickaxe/sword (wood + stone + table + furnace), \
iron_pickaxe/sword (wood + iron + table + furnace)

Survival pressure: The agent has health, food, and thirst meters that \
deplete over time. It must eat (cow, plant) and drink to stay alive, and \
defend against zombies and skeletons that spawn at night. An agent that \
focuses exclusively on crafting without managing survival will die before \
completing harder achievements. Effective play requires balancing survival \
maintenance with technology progression.

22 achievements (evaluated as success rates across eval episodes):
  collect_coal, collect_diamond, collect_drink, collect_iron, collect_sapling, \
collect_stone, collect_wood, defeat_skeleton, defeat_zombie, eat_cow, eat_plant, \
make_iron_pickaxe, make_iron_sword, make_stone_pickaxe, make_stone_sword, \
make_wood_pickaxe, make_wood_sword, place_furnace, place_plant, place_stone, \
place_table, wake_up

The Crafter Score is the geometric mean of (1 + success_rate_i) across all 22 \
achievements. It rewards breadth - an agent must make progress on many \
achievements, not just master a few."""

ROLE_SPECS = {
    "implicit": {
        "role": (
            "You are optimising a prompt used by an LLM to evaluate agent "
            "behaviour segments and assign scalar reward scores (0.0 to 1.0) "
            "during reinforcement learning training. The LLM reads a text "
            "summary of what the agent did over a fixed window of steps and "
            "returns a score. This score is the agent's only reward signal — "
            "it entirely shapes what the agent learns."
        ),
        "constraints": (
            "Keep the same JSON output format (score + reasoning). "
            "Do not change the scoring scale (0.0 to 1.0)."
        ),
    },
    "explicit": {
        "role": (
            "You are optimising a prompt used by an LLM to generate a Python "
            "reward function (compute_reward) that is executed at every step "
            "during reinforcement learning training. The generated function "
            "receives the agent's current and previous step state and returns "
            "a scalar reward. This reward entirely shapes what the agent learns."
        ),
        "constraints": (
            "Do not change the output constraints."
        ),
    },
}