"""Small utilities for reproducible, resumable synthetic experiments."""
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]


def provenance(*filenames):
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "source_sha256": {name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() for name in filenames},
            "data_kind": "synthetic"}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cached_case(path, configuration, evidence):
    path = Path(path)
    if not path.exists():
        return None
    previous = json.loads(path.read_text())
    if previous.get("configuration") == configuration and previous.get("provenance") == evidence:
        return previous
    return None


def relative_error(estimated, actual):
    return float(np.linalg.norm(estimated-actual)/max(np.linalg.norm(actual), 1e-15))


def covariance_error(x, d, truth_x, truth_d):
    """Frobenius error of centered covariance; used only at modest n."""
    x, truth_x = x-x.mean(0), truth_x-truth_x.mean(0)
    def centered_diagonal(diagonal):
        n = len(diagonal)
        return np.diag(diagonal) - (diagonal[:, None]+diagonal[None, :])/n + diagonal.sum()/n**2
    actual = truth_x @ truth_x.T + centered_diagonal(truth_d)
    estimate = x @ x.T + centered_diagonal(d)
    return relative_error(estimate, actual)
