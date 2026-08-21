"""CLI module for DEISM method."""
import os
from .DEISMinterface import DeismMethod


def main() -> None:
    """Run the DEISM method simulation."""
    # JSON path in the uploads folder. This variable is set for the
    # container when it is started up.
    json_file_path = os.environ.get("JSON_PATH")

    print(f"Running DEISM method with JSON_PATH={json_file_path}")
    deism_method_object = DeismMethod(json_file_path)
    deism_method_object.run_simulation()

    # Save the results to a separate file
    deism_method_object.save_results()

    print("DEISM container finished.")
