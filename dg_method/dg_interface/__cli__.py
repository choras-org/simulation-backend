import os
from .DGinterface import DGMethod
import gmsh
import json
import sys

def main() -> None:
    """Run the DG method simulation."""
    # JSON path in the uploads folder. This variable is set for the
    # container when it is started up.
    json_file_path = os.environ.get("JSON_PATH")

    print(f"Running DG method with JSON_PATH={json_file_path}")
    try:
        # Initialize Gmsh before running the simulation
        gmsh.initialize()
        try:
            dg_method_object = DGMethod(json_file_path)
            dg_method_object.run_simulation()
        finally:
            # Ensure that gmsh is finalized
            gmsh.finalize()

    # Outer except to ensure gmsh is finalized above before sys.exit is called
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
    dg_method_object.save_results()

    print("DG container finished.")
