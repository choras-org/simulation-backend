"""Base class implementation of the SimulationMethod interface class."""
from abc import ABC, abstractmethod
from pathlib import Path
import time

import requests


class SimulationMethod(ABC):
    """Abstract base class for simulation methods.

    This class serves as a template for methods required to run a simulation
    and return results to the simulation service executor.

    """

    def __init__(self, input_json_path: str | Path | None):
        """Initialize the simulation method.

        Parameters
        ----------
        input_json_path : str | Path | None, optional
            The path to the input JSON file, by default None

        Raises
        ------
        FileNotFoundError
            If the input JSON file does not exist.

        """
        if input_json_path is None or (
                isinstance(input_json_path, str) and input_json_path == ""):
            raise FileNotFoundError("input_json_path cannot be None or empty")

        input_path = Path(input_json_path)
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input JSON file not found: {input_json_path}")

        self._input_json_path = input_json_path

    @property
    def input_json_path(self) -> str | Path:
        """The input JSON file."""
        return self._input_json_path

    @abstractmethod
    def run_simulation(self):
        """Run the simulation for the given a JSON file."""
        pass

    def save_results(
            self,
            url="http://host.docker.internal:5001/receive",
            max_retries=5,
            delay=2,
        ):
        """Return the results back to the simulation service executor.

        Parameters
        ----------
        url : str, optional
            The URL of the results server,
            by default "http://host.docker.internal:5001/receive" which
            is the default address for local executrion via Docker.
        max_retries : int, optional
            The maximum number of retries if the request fails, by default 5
        delay : int, optional
            The delay in seconds between retries, by default 2

        """

        json_tmp_file = self.input_json_path
        for attempt in range(1, max_retries + 1):
            try:
                with open(json_tmp_file, "rb") as f:
                    response = requests.post(url, files={"file": f})

                if response.status_code == 200:
                    print("Successfully sent file.")
                    return True

                print(
                    f"Attempt {attempt}: ",
                    f"Server returned {response.status_code}")
            except requests.RequestException as exc:
                print(f"Attempt {attempt}: Request failed - {exc}")

            time.sleep(delay)

        print("Max retries reached. Giving up.")
        return False
