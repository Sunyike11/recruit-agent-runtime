from typing import Any, Dict, Iterable, List, Mapping


SUPPORTED_CHECK_TYPES = {
    "required_keys_present",
    "min_count",
    "max_count",
    "equals",
    "contains",
    "status_is",
    "event_type_present",
    "event_type_absent",
    "event_type_count_at_least",
}

_MISSING = object()


def evaluate_checks(
    target: Any,
    checks: Iterable[Dict[str, Any]],
    expected: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    configured = list(checks)
    if not configured and expected:
        configured = [
            {"type": "equals", "path": key, "value": value}
            for key, value in expected.items()
        ]
    return [evaluate_check(target, check) for check in configured]


def evaluate_check(target: Any, check: Dict[str, Any]) -> Dict[str, Any]:
    check_type = check.get("type", "")
    if check_type not in SUPPORTED_CHECK_TYPES:
        return _result(check_type or "<missing>", False, error=f"unsupported check type: {check_type}")
    try:
        if check_type == "required_keys_present":
            return _required_keys_present(target, check)
        if check_type in {"min_count", "max_count"}:
            return _count_check(target, check, check_type)
        if check_type == "equals":
            return _equals(target, check)
        if check_type == "contains":
            return _contains(target, check)
        if check_type == "status_is":
            return _status_is(target, check)
        if check_type in {"event_type_present", "event_type_absent", "event_type_count_at_least"}:
            return _event_type_check(target, check, check_type)
    except Exception as exc:
        return _result(check_type, False, error=str(exc))
    return _result(check_type, False, error="check was not evaluated")


def _required_keys_present(target: Any, check: Dict[str, Any]) -> Dict[str, Any]:
    actual = _path_value(target, check.get("path", ""))
    keys = list(check.get("keys", []))
    if not isinstance(actual, Mapping):
        return _result(
            "required_keys_present",
            False,
            expected=keys,
            actual={"type": type(actual).__name__},
            error="target is not a mapping",
        )
    missing = [key for key in keys if key not in actual]
    return _result(
        "required_keys_present",
        not missing,
        expected=keys,
        actual={"present_keys": sorted(str(key) for key in actual.keys()), "missing_keys": missing},
    )


def _count_check(target: Any, check: Dict[str, Any], check_type: str) -> Dict[str, Any]:
    value = _path_value(target, check.get("path", ""))
    threshold = check.get("value")
    if threshold is None:
        threshold = check.get("min_count" if check_type == "min_count" else "max_count")
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"{check_type} requires numeric value")
    try:
        actual_count = len(value)
    except TypeError:
        return _result(check_type, False, expected=threshold, actual=None, error="target has no length")
    passed = actual_count >= threshold if check_type == "min_count" else actual_count <= threshold
    return _result(check_type, passed, expected=threshold, actual=actual_count)


def _equals(target: Any, check: Dict[str, Any]) -> Dict[str, Any]:
    actual = _path_value(target, check.get("path", ""))
    expected = check.get("value", check.get("expected"))
    return _result("equals", actual == expected, expected=expected, actual=actual)


def _contains(target: Any, check: Dict[str, Any]) -> Dict[str, Any]:
    actual = _path_value(target, check.get("path", ""))
    expected = check.get("value", check.get("expected"))
    try:
        passed = expected in actual
    except TypeError:
        passed = False
    return _result("contains", passed, expected=expected, actual=actual)


def _status_is(target: Any, check: Dict[str, Any]) -> Dict[str, Any]:
    actual = _path_value(target, check.get("path", "status"))
    expected = check.get("value", check.get("expected"))
    return _result("status_is", actual == expected, expected=expected, actual=actual)


def _event_type_check(target: Any, check: Dict[str, Any], check_type: str) -> Dict[str, Any]:
    event_type = check.get("event_type", "")
    event_types = [_event_type(event) for event in _events_from_target(target, check.get("path", ""))]
    actual_count = sum(1 for value in event_types if value == event_type)
    if check_type == "event_type_present":
        expected = "present"
        passed = actual_count >= 1
    elif check_type == "event_type_absent":
        expected = "absent"
        passed = actual_count == 0
    else:
        expected = check.get("value", check.get("min_count", 1))
        if not isinstance(expected, (int, float)):
            raise ValueError("event_type_count_at_least requires numeric value")
        passed = actual_count >= expected
    return _result(check_type, passed, expected=expected, actual=actual_count, event_type=event_type)


def _events_from_target(target: Any, path: str) -> List[Any]:
    candidate = _path_value(target, path) if path else target
    if isinstance(candidate, Mapping) and "events" in candidate:
        candidate = candidate["events"]
    elif hasattr(candidate, "events"):
        candidate = candidate.events
    if isinstance(candidate, (list, tuple)):
        return list(candidate)
    return []


def _event_type(event: Any) -> str:
    if isinstance(event, Mapping):
        return str(event.get("event_type", "") or "")
    return str(getattr(event, "event_type", "") or "")


def _path_value(target: Any, path: str) -> Any:
    if not path:
        return target
    current = target
    for component in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(component, _MISSING)
        else:
            current = getattr(current, component, _MISSING)
        if current is _MISSING:
            return None
    return current


def _result(
    name: str,
    passed: bool,
    expected: Any = None,
    actual: Any = None,
    error: str = "",
    **metadata: Any,
) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "expected": expected,
        "actual": actual,
        "error": error,
        **metadata,
    }
