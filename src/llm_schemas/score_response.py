# JSON schema for structured LLM output in segment evaluation.
# Sent as response_format to the API to constrain output structure.
# Score is a continuous 0-1 float; "reasoning" provides some transparency to LLM reasoning.
segment_score_response = {
    "type": "json_schema",
    "json_schema": {
        "name": "SegmentScoreResponse",
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "reasoning": {
                    "type": "string",
                },
            },
            "required": ["score", "reasoning"],
            "additionalProperties": False,
        },
    },
}