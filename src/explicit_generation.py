"""Offline generation of explicit (EUREKA-style) reward functions via LLM.

This module handles the complete generation pipeline:
1. Load the environment-specific prompt template
2. Call the LLM to generate a compute_reward() Python function
3. Extract the code from the JSON response
4. Validate the code (syntax, function name, parameter signature)
5. On validation failure, re-prompt the LLM with the error (up to 3 attempts)
6. Save the validated code + generation metadata to generated_rewards/

The generated code is a standalone .py file that can be loaded at train-time
by ExplicitRewardModel via importlib. No LLM calls happen during training.
"""

import ast
import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

from config import get_project_root, load_prompt, infer_provider_for_model
from llm_client import create_provider
from llm_schemas.code_response import code_generation_response

logger = logging.getLogger(__name__)

# The exact parameter names the generated function must have.
# Validation checks this against the AST to catch misnamed parameters
# before they cause TypeError crashes during training.
EXPECTED_PARAMS = ("current_step", "prev_step", "terminated", "truncated")

MAX_ATTEMPTS = 3

def extract_code_from_response(response_text: str) -> str:
    """Extract Python code from a JSON-wrapped LLM response.

    The LLM is instructed to return {"code": "<python function>"}.
    json.loads() handles unescaping (\\n -> newline, \\" -> quote, etc.)
    so the extracted string is valid Python source ready to validate and save.

    Args:
        response_text: Raw JSON string from the LLM response.

    Returns:
        The Python code string, stripped of surrounding whitespace.

    Raises:
        ValueError: If the response is not valid JSON or missing the 'code' key.
    """
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Response is not valid JSON: {e}")
    
    if "code" not in data:
        raise ValueError(
            f"Response JSON missing 'code' key. Keys found: {list(data.keys())}"
        )
    
    code = data["code"]
    if not isinstance(code, str):
        raise ValueError(f"'code' value must be a string, got {type(code).__name__}")
    
    return code.strip()


def validate_reward_code(code: str) -> None:
    """Validate that generated code defines compute_reward with the correct signature.

    Performs three checks matching the plan's validation strategy:
      1. Syntax — ast.parse() catches syntax errors
      2. Function exists — the AST must contain a FunctionDef named 'compute_reward'
      3. Signature — parameter names must match EXPECTED_PARAMS exactly

    Args:
        code: Python source code string to validate.

    Raises:
        ValueError: If any validation check fails. The error message is included
            in the retry prompt so the LLM can understand and fix the issue.
    """
    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error on line {e.lineno}: {e.msg}")
    
    # 2. Look for a function named compute_reward in the AST.
    # ast.walk() visits every node in the tree regardless of nesting depth
    # so this finds the function even if the LLM wraps it inside something
    func_defs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "compute_reward"
    ]
    if not func_defs:
        raise ValueError(
            "No function named 'compute_reward' found in generated code"
        )
        
    # 3. Check parameter names match exactly
    # func.args.args is the list of ast.args nodes for positional parameters
    func = func_defs[0]
    param_names = tuple(arg.arg for arg in func.args.args)
    if param_names != EXPECTED_PARAMS:
        raise ValueError(
            f"Wrong parameters: got {param_names}, expected {EXPECTED_PARAMS}"
        )
        

def generate_reward_code(
    env_alias: str,
    llm_model: str,
    prompt_version: str,
    temperature: float,
) -> Path:
    """Generate a reward function via LLM and save to generated_rewards/.

    Makes up to MAX_ATTEMPTS LLM calls. On the first attempt, sends only the
    generation prompt. On retries, appends the previous (failed) response and
    the validation error as conversation context so the LLM can correct itself.

    Args:
        env_alias: Environment name ("crafter" or "minigrid").
        llm_model: Full model identifier (e.g. "openai/gpt-5.2").
        prompt_version: Name of the prompt template file (without .txt extension).
        temperature: LLM sampling temperature.

    Returns:
        Path to the output directory containing reward_fn.py and metadata.yaml.

    Raises:
        RuntimeError: If all MAX_ATTEMPTS attempts fail validation.
    """
    prompt_text = load_prompt(env_alias, prompt_version)
    
    provider_name = infer_provider_for_model(llm_model)
    provider = create_provider(provider_name, llm_model, temperature)
    
    # Build output directory: generated_rewards/{env}_{model}_{timestamp}/
    project_root = get_project_root()
    model_tag = llm_model.replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        project_root / "generated_rewards" / f"{env_alias}_{model_tag}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    
    errors = []
    raw_responses = []
    code = None
    final_usage = None
    
    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info(f"Generation attempt {attempt}/{MAX_ATTEMPTS}")
        
        # On the first attempt, send just the generation prompt.
        # On retries, include the failed response + error as conversation
        # history so the LLM sees what went wrong and can fix it
        if attempt == 1:
            messages = [{"role": "user", "content": prompt_text}]
        else:
            retry_msg = (
                f"Your previous response had an error:\n{errors[-1]}\n\n"
                f"Please fix the issue and return the corrected compute_reward "
                f"function in the same JSON format."
            )
            messages = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": raw_responses[-1]},
                {"role": "user", "content": retry_msg},
            ]
            
        # Call LLM with the code generation response schema.
        # OpenRouter enforces the {"code": "..."} schema server-side.
        # Mistral uses json_object mode (valid JSON, no schema enforcement)
        response = provider.chat_complete(
            messages, response_format=code_generation_response
        )
        raw_responses.append(response.content)
        final_usage = response.usage
        
        # Extract code from JSON, then validate syntax + signature.
        # Both extraction and validation raise ValueError on failure,
        # which gets caught and added to the errors list for the retry prompt
        try:
            extracted = extract_code_from_response(response.content)
            validate_reward_code(extracted)
            code = extracted
            logger.info(f"Validation passed on attempt {attempt}")
            break
        except ValueError as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            errors.append(str(e))
            
    if code is None:
        raise RuntimeError(
            f"All {MAX_ATTEMPTS} generation attempts failed. Errors:\n"
            + "\n".join(
                f"  Attempt {i+1}: {err}" for i, err in enumerate(errors)
            )
        )
        
    # Save the validated reward function as importable .py file
    reward_path = output_dir / "reward_fn.py"
    reward_path.write_text(code + "\n")
    logger.info(f"Saved reward function to {reward_path}")
    
    # Save generation metadata alongside the code for reproducibility
    # raw_responses included so the exact LLM output can be inspected later
    metadata = {
        "env": env_alias,
        "llm_model": llm_model,
        "llm_provider": provider_name,
        "prompt_version": prompt_version,
        "temperature": temperature,
        "timestamp": timestamp,
        "attempts": len(raw_responses),
        "errors": errors if errors else None,
        "raw_responses": raw_responses,
        "usage": {
            "prompt_tokens": final_usage.prompt_tokens,
            "completion_tokens": final_usage.completion_tokens,
            "total_tokens": final_usage.total_tokens,
        } if final_usage else None,
    }
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
        
    logger.info(f"Generation complete: {output_dir}")
    return output_dir