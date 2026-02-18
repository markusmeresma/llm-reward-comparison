from multiprocessing import Value
from environments.minigrid_adapter import MiniGridAdapter
from environments.crafter_adapter import CrafterAdapter
from environments.adapter import EnvAdapter
"""
Map env IDs to adapters.
"""

def get_adapter(env_id: str) -> EnvAdapter:
    if env_id.startswith("MiniGrid"):
        return MiniGridAdapter()
    elif env_id.startswith("Crafter"):
        return CrafterAdapter()
    else:
        raise ValueError(f"No adapter for environment: {env_id}")

