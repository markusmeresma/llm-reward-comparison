# JSON schema for structured LLM output in prompt optimisation.
# Sent as response_format to the API - OpenRouter enforces the schema
# server-side, Mistral falls back to json_object mode (valid JSON, no
# schema enforcement). Both produce parseable {"prompt": "...", "reasoning": "..."} responses.
prompt_optimization_response = {
    "type": "json_schema",
    "json_schema": {
        "name": "PromptOptimizationResponse",
        "schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                },
                "reasoning": {
                    "type": "string",
                },
            },
            "required": ["prompt", "reasoning"],
            "additionalProperties": False,
        },
    },
}