"""
E2E Test Suite Package Initialization
Multimodal Vietnamese-to-H'Mong Speech-to-Speech Translation System.
Conforms to 4-Tier Verification Architecture (Category-Partition, BVA, Pairwise, Scenarios).
"""

from tests.test_utils import (
    CONFIG_PATH,
    PROJECT_ROOT,
    HAS_NUMPY,
    HAS_TORCH,
    HAS_YAML,
    generate_synthetic_audio,
    calculate_rms_dbfs,
    load_project_config,
)

__all__ = [
    "CONFIG_PATH",
    "PROJECT_ROOT",
    "HAS_NUMPY",
    "HAS_TORCH",
    "HAS_YAML",
    "generate_synthetic_audio",
    "calculate_rms_dbfs",
    "load_project_config",
]
