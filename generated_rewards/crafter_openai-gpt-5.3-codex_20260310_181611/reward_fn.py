def compute_reward(current_step, prev_step, terminated, truncated):
    if prev_step is None:
        return 0.0

    reward = 0.0

    cur_inv = current_step.get('inventory', {})
    prev_inv = prev_step.get('inventory', {})
    cur_ach = current_step.get('achievements', {})
    prev_ach = prev_step.get('achievements', {})

    # (4) Survival shaping
    reward += 0.01  # tiny alive bonus

    cur_food = cur_inv.get('food', 0)
    cur_drink = cur_inv.get('drink', 0)
    cur_energy = cur_inv.get('energy', 0)
    cur_health = cur_inv.get('health', 0)

    prev_food = prev_inv.get('food', 0)
    prev_drink = prev_inv.get('drink', 0)
    prev_energy = prev_inv.get('energy', 0)
    prev_health = prev_inv.get('health', 0)

    if cur_food <= 1:
        reward -= 0.8
    if cur_drink <= 1:
        reward -= 0.8
    if cur_energy <= 1:
        reward -= 0.4
    if cur_health <= 1:
        reward -= 1.2

    reward -= 0.3 * max(0, prev_health - cur_health)
    reward += 0.15 * max(0, cur_food - prev_food)
    reward += 0.15 * max(0, cur_drink - prev_drink)
    reward += 0.08 * max(0, cur_energy - prev_energy)
    reward += 0.20 * max(0, cur_health - prev_health)

    if terminated:
        reward -= 25.0

    # (1) New achievement unlock bonuses
    ach_weights = {
        'collect_wood': 3,
        'collect_sapling': 3,
        'collect_drink': 5,
        'eat_plant': 7,
        'eat_cow': 7,
        'place_table': 14,
        'make_wood_pickaxe': 18,
        'make_wood_sword': 5,
        'collect_stone': 12,
        'collect_coal': 14,
        'place_furnace': 22,
        'make_stone_pickaxe': 26,
        'make_stone_sword': 14,
        'place_stone': 6,
        'place_plant': 3,
        'collect_iron': 30,
        'make_iron_pickaxe': 32,
        'make_iron_sword': 24,
        'collect_diamond': 30,
        'defeat_zombie': 14,
        'defeat_skeleton': 18,
        'wake_up': 0,
    }

    # (3A) Stage selection
    if cur_ach.get('place_table', 0) < 1:
        stage = 'A'
    elif cur_ach.get('make_wood_pickaxe', 0) < 1:
        stage = 'B'
    elif cur_ach.get('collect_stone', 0) < 1 or cur_ach.get('collect_coal', 0) < 1:
        stage = 'C'
    elif cur_ach.get('place_furnace', 0) < 1:
        stage = 'D'
    elif cur_ach.get('make_stone_pickaxe', 0) < 1:
        stage = 'E'
    elif cur_ach.get('collect_iron', 0) < 1:
        stage = 'F'
    else:
        stage = 'G'

    next_milestones = {
        'A': {'place_table'},
        'B': {'make_wood_pickaxe'},
        'C': {'collect_stone', 'collect_coal'},
        'D': {'place_furnace'},
        'E': {'make_stone_pickaxe'},
        'F': {'collect_iron'},
        'G': set(),
    }

    for k, w in ach_weights.items():
        if prev_ach.get(k, 0) < 1 and cur_ach.get(k, 0) >= 1:
            mult = 1.2 if k in next_milestones.get(stage, set()) else 1.0
            reward += w * mult

    # (2) Dense shaping with clipped inventory deltas
    targets = {
        'wood': 8,
        'sapling': 3,
        'stone': 12,
        'coal': 8,
        'iron': 8,
        'diamond': 2,
        'wood_pickaxe': 1,
        'stone_pickaxe': 1,
        'iron_pickaxe': 1,
        'wood_sword': 1,
        'stone_sword': 1,
        'iron_sword': 1,
    }
    base_w = {
        'wood': 0.25,
        'sapling': 0.15,
        'stone': 0.40,
        'coal': 0.55,
        'iron': 0.70,
        'diamond': 1.0,
        'wood_pickaxe': 0.9,
        'stone_pickaxe': 0.9,
        'iron_pickaxe': 0.9,
        'wood_sword': 0.4,
        'stone_sword': 0.4,
        'iron_sword': 0.4,
    }

    stage_mult = {
        'A': {'wood': 2.8, 'sapling': 1.2, 'wood_sword': 0.3, 'stone_sword': 0.3, 'iron_sword': 0.3},
        'B': {'wood': 2.2, 'wood_pickaxe': 2.5, 'wood_sword': 0.3, 'stone_sword': 0.3, 'iron_sword': 0.3},
        'C': {'stone': 3.0, 'coal': 3.0, 'wood_sword': 0.3, 'stone_sword': 0.3, 'iron_sword': 0.3},
        'D': {'stone': 4.0, 'coal': 2.0, 'wood_sword': 0.3, 'stone_sword': 0.3, 'iron_sword': 0.3},
        'E': {'stone_pickaxe': 2.5, 'iron': 2.0, 'wood_sword': 0.3, 'stone_sword': 0.3, 'iron_sword': 0.3},
        'F': {'iron': 4.0, 'wood_sword': 0.3, 'stone_sword': 0.3, 'iron_sword': 0.3},
        'G': {'diamond': 3.0, 'wood_sword': 2.0, 'stone_sword': 2.0, 'iron_sword': 2.0, 'iron': 1.2, 'coal': 1.2, 'stone': 1.2},
    }

    sm = stage_mult.get(stage, {})
    for item, t in targets.items():
        p = prev_inv.get(item, 0)
        c = cur_inv.get(item, 0)
        delta = min(c, t) - min(p, t)
        if delta > 0:
            reward += delta * base_w.get(item, 0.0) * sm.get(item, 1.0)

    # (3C) Prerequisite threshold bonuses (one-time by crossing)
    prev_table = prev_ach.get('place_table', 0) >= 1
    prev_wood_pick = prev_ach.get('make_wood_pickaxe', 0) >= 1
    prev_furnace = prev_ach.get('place_furnace', 0) >= 1
    prev_stone_pick = prev_ach.get('make_stone_pickaxe', 0) >= 1

    prev_wood = prev_inv.get('wood', 0)
    cur_wood = cur_inv.get('wood', 0)
    prev_stone = prev_inv.get('stone', 0)
    cur_stone = cur_inv.get('stone', 0)
    prev_coal = prev_inv.get('coal', 0)
    cur_coal = cur_inv.get('coal', 0)
    prev_iron = prev_inv.get('iron', 0)
    cur_iron = cur_inv.get('iron', 0)

    if not prev_table:
        if prev_wood < 2 <= cur_wood:
            reward += 1.0
        if prev_wood < 4 <= cur_wood:
            reward += 0.5

    if not prev_wood_pick:
        if prev_wood < 6 <= cur_wood:
            reward += 0.8

    if not prev_furnace:
        if prev_stone < 4 <= cur_stone:
            reward += 2.0
        if prev_stone < 6 <= cur_stone:
            reward += 1.0
        if prev_stone < 8 <= cur_stone:
            reward += 1.0
        if prev_wood_pick:
            if prev_coal < 1 <= cur_coal:
                reward += 1.0
            if prev_coal < 2 <= cur_coal:
                reward += 0.5
        if (prev_stone < 6 <= cur_stone) and (prev_coal < 1 <= cur_coal):
            reward += 0.8

    if not prev_stone_pick:
        if prev_iron < 1 <= cur_iron:
            reward += 2.0
        if prev_iron < 2 <= cur_iron:
            reward += 1.0

    # (5) Anti-waste / activity
    action = current_step.get('action', 0)
    if action == 0:
        reward -= 0.02

    inv_unchanged = (cur_inv == prev_inv)
    ach_unchanged = (cur_ach == prev_ach)

    if 7 <= action <= 16 and inv_unchanged and ach_unchanged:
        pi = prev_inv
        pa = prev_ach
        prereq = False
        if action == 8:  # place_table
            prereq = pi.get('wood', 0) >= 2
        elif action == 11:  # make_wood_pickaxe
            prereq = pa.get('place_table', 0) >= 1 and pi.get('wood', 0) >= 1
        elif action == 14:  # make_wood_sword
            prereq = pa.get('place_table', 0) >= 1 and pi.get('wood', 0) >= 1
        elif action == 9:  # place_furnace
            prereq = pi.get('stone', 0) >= 4
        elif action == 12 or action == 15:  # stone tools
            prereq = pa.get('place_furnace', 0) >= 1 and pi.get('wood', 0) >= 1 and pi.get('stone', 0) >= 1
        elif action == 13 or action == 16:  # iron tools
            prereq = pa.get('place_furnace', 0) >= 1 and pi.get('wood', 0) >= 1 and pi.get('iron', 0) >= 1
        elif action == 7:  # place_stone
            prereq = pi.get('stone', 0) >= 1
        elif action == 10:  # place_plant
            prereq = pi.get('sapling', 0) >= 1

        if prereq:
            reward -= 0.06

    prev_pos = prev_step.get('pos', (0, 0))
    cur_pos = current_step.get('pos', (0, 0))
    if inv_unchanged and ach_unchanged:
        if cur_pos == prev_pos:
            reward -= 0.01
        else:
            reward += 0.001

    # (6) Episode-end breadth bonus
    if truncated and not terminated:
        count_unlocked = 0
        for k, v in cur_ach.items():
            if k != 'wake_up' and v >= 1:
                count_unlocked += 1
        reward += min(6.0, 0.3 * count_unlocked)

    # (7) Clip final reward
    if reward > 30.0:
        reward = 30.0
    elif reward < -30.0:
        reward = -30.0

    return float(reward)
