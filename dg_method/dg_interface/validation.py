"""Method-specific geometry validation for the DG method (lightweight).

This module provides geometry validation checks specific to the Discontinuous
Galerkin solver. It is imported dynamically by the backend (no heavy deps on
edg-acoustics).
"""


def run_method_validation(input_file: str) -> dict:
    """Validate geometry for DG method.

    Args:
        input_file: Path to the geometry file (e.g., .obj) to validate.

    Returns:
        dict with keys:
            - compatible (bool): True if geometry is suitable, False otherwise
            - reason (str): Human-readable explanation
    """
    # TODO: real checks (watertightness, manifold, planar-faceted, etc.).
    # Hardcoded result for now (incompatible sample to test override):
    return {
        "compatible": False,
        "reason": "DG requires a watertight, manifold, planar-faceted mesh; "
                  "non-planar faces were detected.",
    }
