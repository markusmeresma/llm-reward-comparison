def compute_reward(current_step, prev_step, terminated, truncated):
    # Initialize reward
    reward = 0.0
    
    # Handle first step
    if prev_step is None:
        return reward
    
    # Check for death penalty
    if terminated:
        return -10.0
    
    # Check for time limit penalty
    if truncated:
        return -5.0
    
    # Check for health change
    health_change = current_step['inventory']['health'] - prev_step['inventory']['health']
    if health_change < 0:
        reward += health_change * 2.0
    
    # Check for food and drink consumption
    food_change = current_step['inventory']['food'] - prev_step['inventory']['food']
    drink_change = current_step['inventory']['drink'] - prev_step['inventory']['drink']
    if food_change > 0:
        reward += 0.1
    if drink_change > 0:
        reward += 0.1
    
    # Check for resource collection
    for resource in ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapling']:
        if current_step['inventory'][resource] > prev_step['inventory'][resource]:
            reward += 0.2
    
    # Check for tool crafting
    for tool in ['wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'wood_sword', 'stone_sword', 'iron_sword']:
        if current_step['inventory'][tool] > prev_step['inventory'][tool]:
            reward += 0.5
    
    # Check for achievements unlocked
    for achievement in current_step['achievements']:
        if current_step['achievements'][achievement] > prev_step['achievements'][achievement]:
            reward += 1.0
    
    # Small penalty for noop to encourage activity
    if current_step['action'] == 0:
        reward -= 0.01
    
    return reward
