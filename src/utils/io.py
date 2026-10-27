"""Project paths and JSON serialisation helpers shared by the analysis scripts."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "B2HHH_MagnetDown.root"
# The two magnet polarities of the same data-taking period. The raw asymmetry is
# averaged over them, so a polarity-odd detection asymmetry cancels.
DEFAULT_DATA_PATHS = {
    "MagnetDown": DEFAULT_DATA_PATH,
    "MagnetUp": PROJECT_ROOT / "data" / "raw" / "B2HHH_MagnetUp.root",
}


def display_path(path: Path) -> Path:
    """Return ``path`` relative to the project root when it lies inside it."""
    try:
        return Path(path).relative_to(PROJECT_ROOT)
    except ValueError:
        return Path(path)


def json_safe(value):
    """Convert numpy scalars and non-finite floats into JSON-serialisable values.

    Fit results carry numpy types and can contain ``nan`` or ``inf`` for
    parameters whose uncertainty could not be determined. Those become ``null``
    rather than invalid JSON.
    """
    import numpy as np

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload) -> Path:
    """Write ``payload`` as indented JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")
    return path
