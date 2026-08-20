"""CLI module for deeponet method."""
import os
from .deeponet_interface import DeepONetMethod


def main() -> None:
    """Run the deeponet method simulation."""
    # JSON path in the uploads folder. This variable is set for the
    # container when it is started up.
    json_file_path = os.environ.get("JSON_PATH")

    print(f"Running deeponet method with JSON_PATH={json_file_path}")
    deeponet_method_object = DeepONetMethod(json_file_path)
    deeponet_method_object.run_simulation()

    # Save the results to a separate file
    deeponet_method_object.save_results()

    print("deeponet container finished.")
