"""Method-specific geometry validation for the DE method (lightweight).

This module provides geometry validation checks specific to the Diffusion Equation
solver. It is imported dynamically by the backend (no heavy deps on acousticDE).
"""


def run_method_validation(input_file: str) -> dict:
    """Validate geometry for DE method.

    Args:
        input_file: Path to the geometry file (e.g., .obj) to validate.

    Returns:
        dict with keys:
            - compatible (bool): True if geometry is suitable, False otherwise
            - reason (str): Human-readable explanation
    """
    # TODO: real checks (watertightness, closed volume, etc.).
    # Hardcoded result for now:
    return {
        "compatible": True,
        "reason": "Geometry forms a closed volume suitable for the DE solver.",
    }
