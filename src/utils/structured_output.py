import json
import re
from typing import Any, Dict


class StructuredOutputError(ValueError):
    """Raised when an LLM response cannot be parsed as structured output."""


def extract_json_text(content: str) -> str:
    """Extract a JSON object from plain text or a fenced markdown block."""
    if not content:
        raise StructuredOutputError("Empty content")

    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    object_match = re.search(r"\{.*\}", content, re.DOTALL)
    if object_match:
        return object_match.group(0).strip()

    return content.strip()


def parse_json_object(content: str) -> Dict[str, Any]:
    """Parse an LLM response into a JSON object."""
    json_text = extract_json_text(content)
    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise StructuredOutputError("Expected a JSON object")
    return parsed


def parse_json_object_or_default(content: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a JSON object, returning a copy of default on failure."""
    try:
        parsed = parse_json_object(content)
    except (json.JSONDecodeError, StructuredOutputError, TypeError):
        return default.copy()
    return {**default, **parsed}


def coerce_score(value: Any, default: float = 0) -> float:
    """Convert score-like values to float while keeping bad outputs safe."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
