"""CLI module for MoDART method."""
import os
from .modart_interface import MoDARTMethod
import json
import sys


def main() -> None:
    """Run the MoDART method simulation."""
    # JSON path in the uploads folder. This variable is set for the
    # container when it is started up.
    json_file_path = os.environ.get("JSON_PATH")

    print(f"Running MoDART method with JSON_PATH={json_file_path}")
    modart_method_object = MoDARTMethod(json_file_path)
    try:
        modart_method_object.run_simulation()
    except Exception as e:
        # Write error to result JSON so backend can read it
        with open(json_file_path) as f:
            data = json.load(f)
        data['error'] = {'type': type(e).__name__, 'message': str(e)}
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=4)

        # Ensure the container exits with exit code 1 to indicate failure
        # The status code is used by the backend to determine if the simulation
        # was successful or not.
        sys.exit(1)

    # Save the results to a separate file
    modart_method_object.save_results()

    print("MoDART container finished.")
