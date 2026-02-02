score_response = {
  "type": "json_schema",
  "json_schema": {
    "name": "ScoreResponse",
    "schema": {
      "type": "object",
      "properties": {
        "score": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        }
      },
      "required": ["score"],
      "additionalProperties": False
    }
  }
}
