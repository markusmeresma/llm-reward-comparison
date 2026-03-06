def compute_reward(current_step, prev_step, terminated, truncated):
    reward = 0.0

    # Strong terminal signal for death.
    if terminated:
        reward -= 5.0

    # First step: only terminal handling applies.
    if prev_step is None:
        return float(reward)

    curr_inv = current_step.get('inventory', {})
    prev_inv = prev_step.get('inventory', {})
    curr_ach = current_step.get('achievements', {})
    prev_ach = prev_step.get('achievements', {})

    # One-time (or count-increase) achievement rewards.
    ach_weights = {
        'collect_wood': 0.30,
        'collect_stone': 0.45,
        'collect_coal': 0.55,
        'collect_iron': 0.85,
        'collect_diamond': 3.00,
        'collect_drink': 0.45,
        'collect_sapling': 0.25,
        'eat_cow': 0.70,
        'eat_plant': 0.60,
        'place_table': 1.00,
        'place_furnace': 1.20,
        'place_plant': 0.80,
        'place_stone': 0.35,
        'make_wood_pickaxe': 1.20,
        'make_stone_pickaxe': 1.90,
        'make_iron_pickaxe': 2.60,
        'make_wood_sword': 1.00,
        'make_stone_sword': 1.70,
        'make_iron_sword': 2.30,
        'defeat_zombie': 2.00,
        'defeat_skeleton': 2.50,
        'wake_up': 0.00,
    }

    ach_progress = 0
    for k, w in ach_weights.items():
        d = curr_ach.get(k, 0) - prev_ach.get(k, 0)
        if d > 0:
            ach_progress += d
            reward += w * d

    # Dense shaping for gathering/stocking key resources.
    mat_weights = {
        'wood': 0.03,
        'stone': 0.04,
        'coal': 0.05,
        'iron': 0.08,
        'diamond': 0.20,
        'sapling': 0.03,
    }
    for k, w in mat_weights.items():
        d = curr_inv.get(k, 0) - prev_inv.get(k, 0)
        if d > 0:
            reward += w * d

    # Reward increases in vitals (eating, drinking, sleeping effectively).
    health_d = curr_inv.get('health', 0) - prev_inv.get('health', 0)
    food_d = curr_inv.get('food', 0) - prev_inv.get('food', 0)
    drink_d = curr_inv.get('drink', 0) - prev_inv.get('drink', 0)
    energy_d = curr_inv.get('energy', 0) - prev_inv.get('energy', 0)

    if health_d > 0:
        reward += 0.18 * health_d
    if health_d < 0:
        reward += 0.30 * health_d  # stronger penalty for taking damage

    if food_d > 0:
        reward += 0.05 * food_d
    if drink_d > 0:
        reward += 0.06 * drink_d
    if energy_d > 0:
        reward += 0.04 * energy_d

    # Penalties for critically low vitals to encourage survival behavior.
    health = curr_inv.get('health', 0)
    food = curr_inv.get('food', 0)
    drink = curr_inv.get('drink', 0)
    energy = curr_inv.get('energy', 0)

    if health <= 2:
        reward -= 0.20 * (3 - health)
    if food <= 1:
        reward -= 0.08 * (2 - food)
    if drink <= 1:
        reward -= 0.10 * (2 - drink)
    if energy <= 1:
        reward -= 0.04 * (2 - energy)

    # Tiny incentive for moving/exploring.
    if current_step.get('pos') != prev_step.get('pos'):
        reward += 0.005

    # Discourage idle/no-effect actions a bit.
    action = current_step.get('action', 0)
    if action == 0:
        reward -= 0.01

    # Penalize likely failed craft/place attempts (no observed state change).
    if action in (7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
        any_inv_change = False
        for k in curr_inv:
            if curr_inv.get(k, 0) != prev_inv.get(k, 0):
                any_inv_change = True
                break
        if (not any_inv_change) and ach_progress == 0:
            reward -= 0.03

    # Small survival bonus for reaching time limit alive.
    if truncated and not terminated:
        reward += 0.5

    return float(reward)
