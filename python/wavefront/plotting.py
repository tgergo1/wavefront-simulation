"""Simple plotting-oriented data helpers."""


def snapshot_component_series(snapshot, component: int = 0):
    values = getattr(snapshot, "values", None) or []
    if values:
        return values[component:: max(1, snapshot.components)]

    complex_values = getattr(snapshot, "complex_values", None) or []
    return [value.magnitude() for value in complex_values[component:: max(1, snapshot.components)]]


def snapshot_minmax(snapshot):
    series = snapshot_component_series(snapshot, 0)
    if not series:
        return (0.0, 0.0)
    return (min(series), max(series))


def normalize_series(values):
    if not values:
        return []
    max_abs = max(abs(value) for value in values)
    if max_abs == 0:
        return list(values)
    return [value / max_abs for value in values]
