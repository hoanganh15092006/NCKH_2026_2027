"""
Shared test utilities and fixtures for the E2E Test Suite.
Supports dual-mode execution (host environment and GPU container).
"""

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Environment & Dependency Detection
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import torch
    import torchaudio
    HAS_TORCH = True
except ImportError:
    torch = None
    torchaudio = None
    HAS_TORCH = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False

# Root and Config Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def generate_synthetic_audio(
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    frequency_hz: float = 440.0,
    amplitude: float = 0.5,
    noise_level: float = 0.0
) -> Union[Any, List[float]]:
    """
    Generates synthetic deterministic sine wave audio with optional noise.
    Returns np.ndarray if numpy is installed, otherwise standard float list.
    """
    num_samples = int(duration_s * sample_rate)
    samples = []
    for i in range(num_samples):
        t = i / float(sample_rate)
        pseudo_noise = (math.sin(i * 12345.67) % 1.0 - 0.5) * noise_level if noise_level > 0 else 0.0
        val = amplitude * math.sin(2.0 * math.pi * frequency_hz * t) + pseudo_noise
        val = max(-1.0, min(1.0, val))
        samples.append(val)

    if HAS_NUMPY:
        return np.array(samples, dtype=np.float32)
    return samples


def calculate_rms_dbfs(samples: Union[List[float], Any]) -> float:
    """
    Computes RMS in dBFS from audio samples.
    """
    if HAS_NUMPY and isinstance(samples, np.ndarray):
        rms = float(np.sqrt(np.mean(samples ** 2) + 1e-12))
    else:
        if len(samples) == 0:
            return -100.0
        sum_sq = sum(s ** 2 for s in samples)
        rms = math.sqrt(sum_sq / len(samples) + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)


def parse_yaml_scalar(v: str) -> Any:
    v = v.strip()
    if not v:
        return ""
    if v.startswith("[") and v.endswith("]"):
        # Parse inline list e.g. [8, 8, 2, 2] or ["q_proj", "v_proj"]
        items = [x.strip().strip('"').strip("'") for x in v[1:-1].split(",") if x.strip()]
        parsed_items = []
        for it in items:
            parsed_items.append(parse_yaml_scalar(it))
        return parsed_items
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() in ("null", "none"):
        return None
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v.strip('"').strip("'")


def load_project_config() -> Dict[str, Any]:
    """
    Loads configs/config.yaml using PyYAML if available, or stack-based fallback parser.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

    if HAS_YAML:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    root_config: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root_config)]

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            raw_line = line
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(raw_line) - len(raw_line.lstrip())
            while stack and indent <= stack[-1][0]:
                stack.pop()

            parent = stack[-1][1]

            if ":" in stripped:
                k, v = stripped.split(":", 1)
                k = k.strip()
                v = v.split("#")[0].strip()

                if not v:
                    new_dict: Dict[str, Any] = {}
                    parent[k] = new_dict
                    stack.append((indent, new_dict))
                else:
                    parent[k] = parse_yaml_scalar(v)

    return root_config
