def compute_reward(current_step, prev_step, terminated, truncated):
    if prev_step is None:
        return 0.0

    reward = 0.0

    cur_inv = current_step.get('inventory', {})
    prev_inv = prev_step.get('inventory', {})
    cur_ach = current_step.get('achievements', {})
    prev_ach = prev_step.get('achievements', {})

    cur_pos = current_step.get('pos', (0, 0))
    prev_pos = prev_step.get('pos', (0, 0))
    action = current_step.get('action', 0)

    # (1) Achievement unlock bonuses
    ach_weights = {
        'collect_wood': 3.0,
        'collect_sapling': 3.0,
        'collect_drink': 5.0,
        'eat_plant': 7.0,
        'eat_cow': 7.0,
        'place_table': 14.0,
        'make_wood_pickaxe': 18.0,
        'make_wood_sword': 5.0,
        'collect_stone': 12.0,
        'collect_coal': 14.0,
        'place_furnace': 22.0,
        'make_stone_pickaxe': 26.0,
        'make_stone_sword': 14.0,
        'place_stone': 6.0,
        'place_plant': 3.0,
        'collect_iron': 30.0,
        'make_iron_pickaxe': 32.0,
        'make_iron_sword': 24.0,
        'collect_diamond': 30.0,
        'defeat_zombie': 14.0,
        'defeat_skeleton': 18.0,
        'wake_up': 0.0,
    }

    # Stage selection based on prev achievements (state before current action outcome)
    if prev_ach.get('place_table', 0) < 1:
        stage = 'A'
    elif prev_ach.get('make_wood_pickaxe', 0) < 1:
        stage = 'B'
    elif prev_ach.get('collect_stone', 0) < 1 or prev_ach.get('collect_coal', 0) < 1:
        stage = 'C'
    elif prev_ach.get('place_furnace', 0) < 1:
        stage = 'D'
    elif prev_ach.get('make_stone_pickaxe', 0) < 1:
        stage = 'E'
    elif prev_ach.get('collect_iron', 0) < 1:
        stage = 'F'
    else:
        stage = 'G'

    # Optional sharp curriculum multiplier for the immediate next milestone unlock(s)
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

    # (2) Dense shaping via clipped inventory progress
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

    # (3) Stage multipliers
    mat_mult = {
        'wood': 1.0, 'sapling': 1.0, 'stone': 1.0, 'coal': 1.0, 'iron': 1.0, 'diamond': 1.0
    }
    pickaxe_mult = 1.0
    sword_mult = 1.0

    if stage == 'A':
        mat_mult['wood'] = 2.8
        mat_mult['sapling'] = 1.2
        sword_mult = 0.3
    elif stage == 'B':
        mat_mult['wood'] = 2.2
        pickaxe_mult = 2.5
        sword_mult = 0.3
    elif stage == 'C':
        mat_mult['stone'] = 3.0
        mat_mult['coal'] = 3.0
        sword_mult = 0.3
    elif stage == 'D':
        mat_mult['stone'] = 4.0
        mat_mult['coal'] = 2.0
        sword_mult = 0.3
    elif stage == 'E':
        mat_mult['iron'] = 2.0
        sword_mult = 0.3
    elif stage == 'F':
        mat_mult['iron'] = 4.0
        sword_mult = 0.3
    elif stage == 'G':
        mat_mult['diamond'] = 3.0
        mat_mult['iron'] = 1.2
        mat_mult['coal'] = 1.2
        mat_mult['stone'] = 1.2
        sword_mult = 2.0

    # Dense reward accumulation
    for item, tgt in targets.items():
        prev_v = prev_inv.get(item, 0)
        cur_v = cur_inv.get(item, 0)
        d = min(cur_v, tgt) - min(prev_v, tgt)
        if d > 0:
            w = base_w.get(item, 0.0)
            if item in mat_mult:
                w *= mat_mult[item]
            elif item in ('wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe'):
                local_pick_mult = pickaxe_mult
                if stage == 'E' and item == 'stone_pickaxe':
                    local_pick_mult *= 2.5
                w *= local_pick_mult
            elif item in ('wood_sword', 'stone_sword', 'iron_sword'):
                w *= sword_mult
            reward += w * d

    # (3C) One-time prerequisite threshold bonuses
    prev_wood = prev_inv.get('wood', 0)
    cur_wood = cur_inv.get('wood', 0)
    prev_stone = prev_inv.get('stone', 0)
    cur_stone = cur_inv.get('stone', 0)
    prev_coal = prev_inv.get('coal', 0)
    cur_coal = cur_inv.get('coal', 0)
    prev_iron = prev_inv.get('iron', 0)
    cur_iron = cur_inv.get('iron', 0)

    if prev_ach.get('place_table', 0) < 1:
        if prev_wood < 2 <= cur_wood:
            reward += 1.0
        if prev_wood < 4 <= cur_wood:
            reward += 0.5

    if prev_ach.get('make_wood_pickaxe', 0) < 1:
        if prev_wood < 6 <= cur_wood:
            reward += 0.8

    if prev_ach.get('place_furnace', 0) < 1:
        if prev_stone < 4 <= cur_stone:
            reward += 2.0
        if prev_stone < 6 <= cur_stone:
            reward += 1.0
        if prev_stone < 8 <= cur_stone:
            reward += 1.0
        if prev_ach.get('make_wood_pickaxe', 0) >= 1:
            if prev_coal < 1 <= cur_coal:
                reward += 1.0
            if prev_coal < 2 <= cur_coal:
                reward += 0.5
        if prev_stone < 6 <= cur_stone and prev_coal < 1 <= cur_coal:
            reward += 0.8

    if prev_ach.get('make_stone_pickaxe', 0) < 1:
        if prev_iron < 1 <= cur_iron:
            reward += 2.0
        if prev_iron < 2 <= cur_iron:
            reward += 1.0

    # (4) Survival shaping
    reward += 0.01

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

    food_up = cur_food - prev_food
    drink_up = cur_drink - prev_drink
    energy_up = cur_energy - prev_energy
    health_up = cur_health - prev_health

    if food_up > 0:
        reward += 0.15 * food_up
    if drink_up > 0:
        reward += 0.15 * drink_up
    if energy_up > 0:
        reward += 0.08 * energy_up
    if health_up > 0:
        reward += 0.20 * health_up

    if terminated:
        reward -= 25.0

    # (5) Anti-waste / activity
    if action == 0:
        reward -= 0.02

    inv_keys = [
        'health', 'food', 'drink', 'energy',
        'wood', 'stone', 'coal', 'iron', 'diamond', 'sapling',
        'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
        'wood_sword', 'stone_sword', 'iron_sword'
    ]

    inv_unchanged = True
    for k in inv_keys:
        if cur_inv.get(k, 0) != prev_inv.get(k, 0):
            inv_unchanged = False
            break

    ach_keys = [
        'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron', 'collect_sapling', 'collect_stone', 'collect_wood',
        'defeat_skeleton', 'defeat_zombie', 'eat_cow', 'eat_plant',
        'make_iron_pickaxe', 'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword', 'make_wood_pickaxe', 'make_wood_sword',
        'place_furnace', 'place_plant', 'place_stone', 'place_table', 'wake_up'
    ]

    ach_unchanged = True
    for k in ach_keys:
        if cur_ach.get(k, 0) != prev_ach.get(k, 0):
            ach_unchanged = False
            break

    pos_unchanged = (cur_pos == prev_pos)

    if action in (7, 8, 9, 10, 11, 12, 13, 14, 15, 16) and inv_unchanged and ach_unchanged:
        had_prereq = False
        if action == 8:  # place_table
            had_prereq = prev_inv.get('wood', 0) >= 2
        elif action == 11 or action == 14:  # wood tools
            had_prereq = prev_ach.get('place_table', 0) >= 1 and prev_inv.get('wood', 0) >= 1
        elif action == 9:  # place_furnace
            had_prereq = prev_inv.get('stone', 0) >= 4
        elif action == 12 or action == 15:  # stone tools
            had_prereq = (
                prev_ach.get('place_furnace', 0) >= 1 and
                prev_inv.get('wood', 0) >= 1 and
                prev_inv.get('stone', 0) >= 1
            )
        elif action == 13 or action == 16:  # iron tools
            had_prereq = (
                prev_ach.get('place_furnace', 0) >= 1 and
                prev_inv.get('wood', 0) >= 1 and
                prev_inv.get('iron', 0) >= 1
            )
        elif action == 7:  # place_stone
            had_prereq = prev_inv.get('stone', 0) >= 1
        elif action == 10:  # place_plant
            had_prereq = prev_inv.get('sapling', 0) >= 1

        if had_prereq:
            reward -= 0.06

    if inv_unchanged and ach_unchanged:
        if pos_unchanged:
            reward -= 0.01
        else:
            reward += 0.001

    # (6) Episode-end breadth bonus
    if truncated and not terminated:
        unlocked = 0
        for k in ach_keys:
            if k != 'wake_up' and cur_ach.get(k, 0) >= 1:
                unlocked += 1
        breadth_bonus = 0.3 * unlocked
        if breadth_bonus > 6.0:
            breadth_bonus = 6.0
        reward += breadth_bonus

    # (7) Clip final reward
    if reward > 30.0:
        reward = 30.0
    elif reward < -30.0:
        reward = -30.0

    return float(reward)
