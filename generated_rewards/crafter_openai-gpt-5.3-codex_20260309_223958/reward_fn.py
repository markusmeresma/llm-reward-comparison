def compute_reward(current_step, prev_step, terminated, truncated):
    """
    Args:
        current_step: dict with keys described below
        prev_step: dict with same keys, or None on the first step of an episode
        terminated: bool, True if the agent died (health reached 0)
        truncated: bool, True if the episode hit the time limit

    Returns:
        float: scalar reward for this step
    """
    if prev_step is None:
        return 0.0

    reward = 0.0

    cur_inv = current_step.get("inventory", {})
    prev_inv = prev_step.get("inventory", {})
    cur_ach = current_step.get("achievements", {})
    prev_ach = prev_step.get("achievements", {})

    def inv(key):
        return cur_inv.get(key, 0)

    def pinv(key):
        return prev_inv.get(key, 0)

    def unlocked(name):
        return prev_ach.get(name, 0) < 1 and cur_ach.get(name, 0) >= 1

    # 1) Large sparse rewards for newly unlocked achievements
    ach_reward = {
        "collect_wood": 3.0,
        "collect_sapling": 3.0,
        "collect_drink": 3.0,
        "eat_plant": 4.0,
        "eat_cow": 4.0,
        "place_table": 8.0,
        "make_wood_pickaxe": 10.0,
        "make_wood_sword": 6.0,
        "collect_stone": 8.0,
        "collect_coal": 8.0,
        "place_furnace": 12.0,
        "make_stone_pickaxe": 14.0,
        "make_stone_sword": 10.0,
        "place_stone": 6.0,
        "place_plant": 6.0,
        "collect_iron": 16.0,
        "make_iron_pickaxe": 20.0,
        "make_iron_sword": 16.0,
        "collect_diamond": 30.0,
        "defeat_zombie": 12.0,
        "defeat_skeleton": 14.0,
        "wake_up": 0.0,
    }

    unlock_count = 0
    for k, w in ach_reward.items():
        if unlocked(k):
            reward += w
            unlock_count += 1

    # 2) Dense shaping from clipped inventory progress (anti-farming via caps)
    targets = {
        "wood": 6,
        "sapling": 3,
        "stone": 6,
        "coal": 4,
        "iron": 4,
        "diamond": 2,
    }
    inv_weights = {
        "wood": 0.30,
        "sapling": 0.25,
        "stone": 0.45,
        "coal": 0.55,
        "iron": 0.80,
        "diamond": 1.20,
    }

    material_progress = 0.0
    for k, cap in targets.items():
        d = min(inv(k), cap) - min(pinv(k), cap)
        if d > 0:
            g = d * inv_weights[k]
            reward += g
            material_progress += g

    # Tools/swords: reward first acquisition only
    tools = {
        "wood_pickaxe": 1.5,
        "stone_pickaxe": 2.5,
        "iron_pickaxe": 3.5,
        "wood_sword": 1.0,
        "stone_sword": 2.0,
        "iron_sword": 3.0,
    }
    tool_progress = 0.0
    for k, w in tools.items():
        d = min(inv(k), 1) - min(pinv(k), 1)
        if d > 0:
            g = d * w
            reward += g
            tool_progress += g

    # 3) Curriculum bias toward next missing milestone
    stage_bonus = 0.0
    if prev_ach.get("place_table", 0) < 1:
        wood_gain = max(0, min(inv("wood"), 6) - min(pinv("wood"), 6))
        stage_bonus += 0.60 * wood_gain
        if unlocked("place_table"):
            stage_bonus += 3.0
    elif prev_ach.get("make_wood_pickaxe", 0) < 1:
        wood_gain = max(0, min(inv("wood"), 6) - min(pinv("wood"), 6))
        stage_bonus += 0.40 * wood_gain
        if unlocked("make_wood_pickaxe"):
            stage_bonus += 4.0
    elif prev_ach.get("collect_stone", 0) < 1 or prev_ach.get("collect_coal", 0) < 1:
        stone_gain = max(0, min(inv("stone"), 6) - min(pinv("stone"), 6))
        coal_gain = max(0, min(inv("coal"), 4) - min(pinv("coal"), 4))
        stage_bonus += 0.80 * stone_gain + 0.90 * coal_gain
        if unlocked("collect_stone"):
            stage_bonus += 2.0
        if unlocked("collect_coal"):
            stage_bonus += 2.0
    elif prev_ach.get("place_furnace", 0) < 1:
        stone_gain = max(0, min(inv("stone"), 6) - min(pinv("stone"), 6))
        stage_bonus += 0.50 * stone_gain
        if unlocked("place_furnace"):
            stage_bonus += 4.0
    elif prev_ach.get("make_stone_pickaxe", 0) < 1:
        iron_gain = max(0, min(inv("iron"), 4) - min(pinv("iron"), 4))
        stage_bonus += 0.80 * iron_gain
        if unlocked("make_stone_pickaxe"):
            stage_bonus += 5.0
    elif prev_ach.get("collect_iron", 0) < 1:
        iron_gain = max(0, min(inv("iron"), 4) - min(pinv("iron"), 4))
        stage_bonus += 1.00 * iron_gain
        if unlocked("collect_iron"):
            stage_bonus += 4.0
    else:
        diamond_gain = max(0, min(inv("diamond"), 2) - min(pinv("diamond"), 2))
        stage_bonus += 2.0 * diamond_gain
        if unlocked("collect_diamond"):
            stage_bonus += 4.0
        if unlocked("defeat_zombie"):
            stage_bonus += 2.0
        if unlocked("defeat_skeleton"):
            stage_bonus += 2.5

    reward += stage_bonus

    # 4) Survival shaping
    if terminated:
        reward -= 25.0
    else:
        reward += 0.01

    food = inv("food")
    drink = inv("drink")
    energy = inv("energy")

    if food <= 1:
        reward -= 0.8
    if drink <= 1:
        reward -= 0.8
    if energy <= 1:
        reward -= 0.6

    # Penalize damage, reward recovery (small)
    health_drop = max(0, pinv("health") - inv("health"))
    if health_drop > 0:
        reward -= 0.6 * health_drop

    for k, w in (("food", 0.12), ("drink", 0.12), ("energy", 0.10), ("health", 0.18)):
        inc = inv(k) - pinv(k)
        if inc > 0:
            reward += w * inc

    # 5) Anti-waste / activity
    action = current_step.get("action", 0)
    if action == 0:
        reward -= 0.02

    # Penalize failed place/craft attempts when both achievements and inventory did not change
    if 7 <= action <= 16:
        ach_same = True
        keys_ach = set(prev_ach.keys()) | set(cur_ach.keys())
        for k in keys_ach:
            if prev_ach.get(k, 0) != cur_ach.get(k, 0):
                ach_same = False
                break

        inv_same = True
        keys_inv = set(prev_inv.keys()) | set(cur_inv.keys())
        for k in keys_inv:
            if prev_inv.get(k, 0) != cur_inv.get(k, 0):
                inv_same = False
                break

        if ach_same and inv_same:
            reward -= 0.20

    # Tiny movement bonus only when no substantive progress happened
    pos_prev = prev_step.get("pos", None)
    pos_cur = current_step.get("pos", None)
    if pos_prev is not None and pos_cur is not None and pos_prev != pos_cur:
        if unlock_count == 0 and material_progress <= 0 and tool_progress <= 0:
            reward += 0.001

    # 6) Stability clipping
    if reward > 30.0:
        reward = 30.0
    elif reward < -30.0:
        reward = -30.0

    return float(reward)
