"""CLI module for PFFDTD method."""
import os
from .PFFDTDinterface import PFFDTDMethod


def main() -> None:
    """Run the PFFDTD method simulation."""
    json_file_path = os.environ.get("JSON_PATH")

    print(f"Running PFFDTD method with JSON_PATH={json_file_path}")
    pffdtd_method_object = PFFDTDMethod(json_file_path)
    pffdtd_method_object.run_simulation()
    pffdtd_method_object.save_results()

    print("PFFDTD container finished.")
