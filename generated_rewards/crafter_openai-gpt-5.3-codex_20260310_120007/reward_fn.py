def compute_reward(current_step, prev_step, terminated, truncated):
    if prev_step is None:
        return 0.0

    reward = 0.0

    cur_inv = current_step.get('inventory', {})
    prev_inv = prev_step.get('inventory', {})
    cur_ach = current_step.get('achievements', {})
    prev_ach = prev_step.get('achievements', {})

    # (1) Big bonuses for first-time achievement unlocks
    ach_weights = {
        'collect_wood': 3.0,
        'collect_sapling': 3.0,
        'collect_drink': 4.0,
        'eat_plant': 6.0,
        'eat_cow': 6.0,
        'place_table': 10.0,
        'make_wood_pickaxe': 12.0,
        'make_wood_sword': 8.0,
        'collect_stone': 10.0,
        'collect_coal': 12.0,
        'place_furnace': 18.0,
        'make_stone_pickaxe': 22.0,
        'make_stone_sword': 14.0,
        'place_stone': 6.0,
        'place_plant': 4.0,
        'collect_iron': 26.0,
        'make_iron_pickaxe': 28.0,
        'make_iron_sword': 22.0,
        'collect_diamond': 30.0,
        'defeat_zombie': 14.0,
        'defeat_skeleton': 18.0,
        'wake_up': 0.0,
    }
    for k, w in ach_weights.items():
        if prev_ach.get(k, 0) <= 0 and cur_ach.get(k, 0) >= 1:
            reward += w

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

    # (2) Dense shaping: clipped inventory progress
    targets = {
        'wood': 8, 'sapling': 3, 'stone': 8, 'coal': 6, 'iron': 6, 'diamond': 2,
        'wood_pickaxe': 1, 'stone_pickaxe': 1, 'iron_pickaxe': 1,
        'wood_sword': 1, 'stone_sword': 1, 'iron_sword': 1,
    }
    base_w = {
        'wood': 0.25, 'sapling': 0.15, 'stone': 0.35, 'coal': 0.45, 'iron': 0.6, 'diamond': 1.0,
        'wood_pickaxe': 0.8, 'stone_pickaxe': 0.8, 'iron_pickaxe': 0.8,
        'wood_sword': 0.5, 'stone_sword': 0.5, 'iron_sword': 0.5,
    }

    multipliers = {}
    for k in targets:
        multipliers[k] = 1.0

    # (3B) Stage multipliers
    if stage == 'A':
        multipliers['wood'] = 2.5
        multipliers['sapling'] = 1.2
    elif stage == 'B':
        multipliers['wood'] = 2.0
        multipliers['wood_pickaxe'] = 2.0
    elif stage == 'C':
        multipliers['stone'] = 2.5
        multipliers['coal'] = 2.5
    elif stage == 'D':
        multipliers['stone'] = 3.0
        multipliers['coal'] = 1.5
    elif stage == 'E':
        multipliers['iron'] = 2.5
        multipliers['stone_pickaxe'] = 2.0
        multipliers['coal'] = 1.5
        multipliers['stone'] = 1.5
    elif stage == 'F':
        multipliers['iron'] = 3.0
        multipliers['iron_pickaxe'] = 1.5
    else:  # G
        multipliers['diamond'] = 2.5
        multipliers['iron_pickaxe'] = 1.5
        multipliers['iron_sword'] = 1.8
        multipliers['stone_sword'] = 1.2
        multipliers['iron'] = 1.5

    for k, t in targets.items():
        p = prev_inv.get(k, 0)
        c = cur_inv.get(k, 0)
        d = min(c, t) - min(p, t)
        if d > 0:
            reward += d * base_w[k] * multipliers.get(k, 1.0)

    # (3C) Prerequisite threshold bonuses
    prev_wood = prev_inv.get('wood', 0)
    cur_wood = cur_inv.get('wood', 0)
    if prev_ach.get('place_table', 0) < 1:
        if prev_wood < 2 <= cur_wood:
            reward += 1.0
        if prev_wood < 4 <= cur_wood:
            reward += 0.5

    prev_stone = prev_inv.get('stone', 0)
    cur_stone = cur_inv.get('stone', 0)
    if prev_ach.get('place_furnace', 0) < 1:
        if prev_stone < 4 <= cur_stone:
            reward += 2.0
        if prev_stone < 6 <= cur_stone:
            reward += 1.0

    prev_iron = prev_inv.get('iron', 0)
    cur_iron = cur_inv.get('iron', 0)
    if prev_ach.get('make_stone_pickaxe', 0) < 1:
        if prev_iron < 1 <= cur_iron:
            reward += 2.0
        if prev_iron < 2 <= cur_iron:
            reward += 1.0

    if (prev_ach.get('make_wood_pickaxe', 0) >= 1 or cur_ach.get('make_wood_pickaxe', 0) >= 1):
        if prev_inv.get('coal', 0) < 1 <= cur_inv.get('coal', 0):
            reward += 1.0

    # (4) Survival shaping
    reward += 0.01  # alive bonus

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

    health_drop = prev_health - cur_health
    if health_drop > 0:
        reward -= 0.3 * health_drop

    food_gain = cur_food - prev_food
    drink_gain = cur_drink - prev_drink
    energy_gain = cur_energy - prev_energy
    health_gain = cur_health - prev_health

    if food_gain > 0:
        reward += 0.15 * food_gain
    if drink_gain > 0:
        reward += 0.15 * drink_gain
    if energy_gain > 0:
        reward += 0.08 * energy_gain
    if health_gain > 0:
        reward += 0.2 * health_gain

    if terminated:
        reward -= 25.0

    # (5) Anti-waste / activity
    action = current_step.get('action', 0)
    if action == 0:
        reward -= 0.02

    inv_unchanged = (cur_inv == prev_inv)
    ach_unchanged = (cur_ach == prev_ach)

    if 7 <= action <= 16 and inv_unchanged and ach_unchanged:
        reward -= 0.07

    pos_same = current_step.get('pos', None) == prev_step.get('pos', None)
    if inv_unchanged and ach_unchanged:
        if pos_same:
            reward -= 0.01
        else:
            reward += 0.001

    # (6) Episode-end breadth bonus
    if truncated and not terminated:
        count_unlocked = 0
        for k, v in cur_ach.items():
            if k != 'wake_up' and v >= 1:
                count_unlocked += 1
        breadth_bonus = 0.3 * count_unlocked
        if breadth_bonus > 6.0:
            breadth_bonus = 6.0
        reward += breadth_bonus

    # (7) Clip final reward
    if reward > 30.0:
        reward = 30.0
    elif reward < -30.0:
        reward = -30.0

    return float(reward)
