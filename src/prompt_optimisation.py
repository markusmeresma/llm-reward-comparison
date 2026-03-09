import csv
from pathlib import Path
import yaml
import json
import logging
from datetime import datetime

from config import get_project_root, infer_provider_for_model
from llm_client import create_provider
from llm_schemas.prompt_response import prompt_optimization_response
from optimiser_prompts import CRAFTER_ENV_CONTEXT, ROLE_SPECS

logger = logging.getLogger(__name__)

OPTIMISER_MODEL = "openai/gpt-5.2"

def optimise_prompt(
    env_alias: str,
    reward_type: str,
    experiment_path: Path,
    current_prompt_path: Path,
    history_path: Path,
    temperature: float
) -> Path:
    """Run one iteration of prompt optimisation.
    
    Reads eval metrics, builds the optimizer prompt, calls the LLM,
    saves the revised prompt + metadata, and updates history.
    
    Returns:
        Path to the output directory containing prompt.txt and metadata.yaml.
    """
    # 1. Read eval metrics from the completed experiment
    metrics = read_eval_metrics(experiment_path)
    
    # 2. Load the current prompt text
    current_prompt_text = current_prompt_path.read_text().strip()
    
    # 3. Load history (empty list on first iter)
    history = load_history(history_path)
    
    # 4. Get previous metrics for delta comparison (if any)
    previous_metrics = None
    if history:
        prev_experiment = history[-1]["experiment_path"]
        try:
            previous_metrics = read_eval_metrics(Path(prev_experiment))
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not load previous metrics for delta comparison: {e}")
            
    # 5. Format metrics and history into text
    metrics_text = format_metrics_for_optimizer(metrics, previous_metrics)
    history_text = format_history_for_optimizer(history)
    
    # 6. Build the full optimiser prompt
    prompt = build_optimiser_prompt(
        reward_type=reward_type,
        env_alias=env_alias,
        current_prompt_text=current_prompt_text,
        metrics_text=metrics_text,
        history_text=history_text,
    )
    
    # 7. Call the optimiser LLM
    provider_name = infer_provider_for_model(OPTIMISER_MODEL)
    provider = create_provider(provider_name, OPTIMISER_MODEL, temperature)
    
    messages = [{"role": "user", "content": prompt}]
    logger.info(f"Calling {OPTIMISER_MODEL} for prompt optimisation...")
    response = provider.chat_complete(messages, response_format=prompt_optimization_response)
    
    # 8. Parse JSON response
    try:
        data = json.loads(response.content)
        revised_prompt = data["prompt"]
        reasoning = data["reasoning"]
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(
            f"Failed to parse optimiser response: {e}\n"
            f"Raw response: {response.content[:500]}"
        )
    
    # 9. Determine iteration number and create output directory
    iteration = len(history) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = get_project_root()
    output_dir = (
        project_root / "optimised_prompts" / f"{env_alias}_{reward_type}"
        / f"iter{iteration}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 10. Save prompt.txt and metadata.yaml
    (output_dir / "prompt.txt").write_text(revised_prompt + "\n")
    
    metadata = {
        "optimiser_model": OPTIMISER_MODEL,
        "iteration": iteration,
        "experiment_path": str(experiment_path),
        "input_prompt_path": str(current_prompt_path),
        "pre_optimisation_crafter_score": metrics["crafter_score"],
        "pre_optimisation_metrics": metrics,
        "reasoning": reasoning,
        "raw_response": response.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        } if response.usage else None,
    }
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
        
    # 11. Append to history
    prompt_path = str(output_dir / "prompt.txt")
    append_history(history_path, {
        "iteration": iteration,
        "prompt_path": prompt_path,
        "experiment_path": str(experiment_path),
        "pre_optimisation_crafter_score": metrics["crafter_score"],
        "pre_optimisation_metrics": metrics,
        "reasoning_summary": reasoning,
    })
    
    logger.info(f"Optimisation complete (iteration {iteration}): {output_dir}")
    return output_dir

def build_optimiser_prompt(
    reward_type: str,
    env_alias: str,
    current_prompt_text: str,
    metrics_text: str,
    history_text: str,
) -> str:
    """Assemble the full optimizer prompt from fixed and variable parts.
    
    Args:
        reward_type: 'implicit' or 'explicit'
        env_alias: 'crafter' (only supported env for now)
        current_prompt_text: the full text of the prompt being optimised
        metrics_text: output of format_metrics_for_optimizer()
        history_text: output of format_history_for_optimizer() (empty string if first iteration)
    """
    if env_alias != "crafter":
        raise ValueError(f"Prompt optimisation only supports crafter, got: {env_alias}")
    if reward_type not in ROLE_SPECS:
        raise ValueError(f"Unknown reward_type: {reward_type}. Must be 'implicit' or 'explicit'.")
    
    spec = ROLE_SPECS[reward_type]
    
    sections = []
    
    # 1. Role
    sections.append(spec["role"])
    
    # 2. Environment context
    sections.append(f"=== ENVIRONMENT ===\n\n{CRAFTER_ENV_CONTEXT}")
    
    # 3. Current prompt being optimised
    sections.append(
        f"=== CURRENT PROMPT (to be revised) ===\n\n{current_prompt_text}"
    )
    
    # 4. Performance metrics
    sections.append(metrics_text)
    
    # 5. Optimisation history
    if history_text:
        sections.append(history_text)
        
    sections.append(
        f"=== TASK ===\n\n"
        f"Analyse the performance metrics above and revise the prompt to "
        f"improve the agent's Crafter Score.\n\n"
        f"Think step by step:\n"
        f"1. Diagnose which achievements are failing and why\n"
        f"2. Hypothesise what specific prompt changes would address the "
        f"failures without breaking what already works\n"
        f"3. Write the complete revised prompt\n\n"
        f"Constraints: {spec['constraints']}\n\n"
        f'Respond as JSON: {{"prompt": "<the complete revised prompt>", '
        f'"reasoning": "<your diagnosis and rationale>"}}'
    )
    
    return "\n\n".join(sections)

def load_history(history_path: Path) -> list[dict]:
    """Read history.yaml, returning a list of iteration summaries.
    
    Returns an empty list if the file doesn't exist yet (first iteration).
    """
    if not history_path.exists():
        return []
    
    data = yaml.safe_load(history_path.read_text())
    return data if data else []


def append_history(history_path: Path, iteration_summary: dict) -> None:
    """Append one iteration summary to history.yaml.
    
    Creates the file and parent directories if they don't exist.
    """
    history = load_history(history_path)
    history.append(iteration_summary)
    
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        yaml.dump(history, f, default_flow_style=False, sort_keys=False)


def format_history_for_optimizer(history: list[dict]) -> str:
    """Convert history list into a text block for the optimizer prompt context.
    
    Returns empty string if history is empty (first iteration).
    """
    if not history:
        return ""
    
    lines = ["=== OPTIMISATION HISTORY ===", ""]
    
    for entry in history:
        lines.append(f"--- Iteration {entry['iteration']} ---")
        lines.append(f"Crafter Score: {entry['pre_optimisation_crafter_score']:.2f}")
        
        if "pre_optimisation_metrics" in entry:
            for metric_name, value in entry["pre_optimisation_metrics"].items():
                lines.append(f"  {metric_name}: {value}")
                
        lines.append(f"Reasoning: {entry.get('reasoning_summary', 'N/A')}")
        lines.append("")
        
    return "\n".join(lines)

def read_eval_metrics(experiment_path: Path) -> dict:
    """Read the final row of crafter_eval_metrics.csv from an experiment directory.
    
    Returns a dict with string keys and float values for: crafter_score,
    mean_achievements_per_episode, and all success_rate_* columns.
    Raises FileNotFoundError if the CSV is missing, ValueError if it's empty.
    """
    csv_path = experiment_path / "crafter_eval_metrics.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Eval metrics CSV not found: {csv_path}"
        )
        
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    if not rows:
        raise ValueError(
            f"Eval metrics CSV is empty (no data rows): {csv_path}"
        )
        
    last_row = rows[-1]
    
    # Extract the columns the optimizer needs - aggregate scores plus
    # all 22 per-achievement success rates. Skip bookkeeping columns
    # (timestep, n_eval_episodes) that aren't useful for prompt diagnosis.
    metrics = {}
    for key, value in last_row.items():  
        if key in ("crafter_score", "mean_achievements_per_episode") or key.startswith("success_rate_"):
            metrics[key] = float(value)
    
    return metrics


def format_metrics_for_optimizer(
    current_metrics: dict, previous_metrics: dict | None = None
) -> str:
    """Format metrics into a readable text block for the optimizer prompt.
    
    Lists all achievement success rates with deltas where previous metrics exist.
    """
    lines = ["=== PERFORMANCE METRICS ===", ""]
    
    score = current_metrics["crafter_score"]
    mean_ach = current_metrics["mean_achievements_per_episode"]
    
    if previous_metrics:
        prev_score = previous_metrics["crafter_score"]
        prev_mean = previous_metrics["mean_achievements_per_episode"]
        lines.append(f"Crafter Score: {score:.2f} ({_delta_str(score - prev_score)})")
        lines.append(f"Mean Achievements Per Episode: {mean_ach:.2f} ({_delta_str(mean_ach - prev_mean)})")
    else:
        lines.append(f"Crafter Score: {score:.2f}")
        lines.append(f"Mean Achievements Per Episode: {mean_ach:.2f}")
        
    lines.append("")
    lines.append("--- Achievement Success Rates ---")
    
    for key, rate in sorted(current_metrics.items()):
        if not key.startswith("success_rate_"):
            continue
        name = key.removeprefix("success_rate_")
        entry = f"  {name}: {rate:.1f}%"
        if previous_metrics and key in previous_metrics:
            delta = rate - previous_metrics[key]
            if delta != 0:
                entry += f"  [{_delta_str(delta)}]"
        lines.append(entry)
        
    return "\n".join(lines)
        

def _delta_str(delta: float) -> str:
    """Format a numeric delta as a signed string with direction label."""
    if delta > 0:
        return f"+{delta:.2f} improved"
    elif delta < 0:
        return f"{delta:.2f} declined"
    else:
        return "unchanged"
    