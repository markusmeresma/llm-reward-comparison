# JSON schema for structured LLM output in code generation.
# Sent as response_format to the API - OpenRouter enforces the schema
# server-side, Mistral falls back to json_object mode (valid JSON, no
# schema enforcement). Both produce parseable {"code": "..."} responses.
code_generation_response = {
    "type": "json_schema",
    "json_schema": {
        "name": "CodeGenerationResponse",
        "schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}