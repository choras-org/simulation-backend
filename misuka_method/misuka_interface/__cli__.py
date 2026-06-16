"""CLI module for misuka method."""
import os
from .misuka_interface import misukaMethod


def main() -> None:
    """Run the misuka method simulation."""
    # JSON path in the uploads folder. This variable is set for the
    # container when it is started up.
    json_file_path = os.environ.get("JSON_PATH")

    print(f"Running misuka method with JSON_PATH={json_file_path}")
    misuka_method_object = misukaMethod(json_file_path)
    misuka_method_object.run_simulation()

    # Save the results to a separate file
    misuka_method_object.save_results()

    print("misuka container finished.")
