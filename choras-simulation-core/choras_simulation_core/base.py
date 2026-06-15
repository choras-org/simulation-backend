"""Base class for CHORAS simulation methods.

This module provides the abstract base class that all simulation methods
must inherit from. It handles common functionality like JSON file validation
and result transmission back to the backend.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path

import requests


class SimulationMethod(ABC):
    """Abstract base class for simulation methods.

    This class serves as a template for methods required to run a simulation
    and return results to the simulation service executor.

    Parameters
    ----------
    input_json_path : str | Path | None
        The path to the input JSON file containing simulation configuration.

    Raises
    ------
    FileNotFoundError
        If the input JSON file does not exist or is None/empty.

    Examples
    --------
    >>> class MySimulationMethod(SimulationMethod):
    ...     def run_simulation(self):
    ...         # Implementation here
    ...         pass
    >>> method = MySimulationMethod("/path/to/config.json")
    >>> method.run_simulation()
    >>> method.save_results()

    """

    def __init__(self, input_json_path: str | Path | None):
        """Initialize the simulation method.

        Parameters
        ----------
        input_json_path : str | Path | None
            The path to the input JSON file. Cannot be None or empty.

        Raises
        ------
        FileNotFoundError
            If the input JSON file does not exist or path is None/empty.

        """
        if input_json_path is None or (
            isinstance(input_json_path, str) and input_json_path == ""
        ):
            raise FileNotFoundError("input_json_path cannot be None or empty")

        input_path = Path(input_json_path)
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input JSON file not found: {input_json_path}"
            )

        self._input_json_path = input_json_path

    @property
    def input_json_path(self) -> str | Path:
        """Get the input JSON file path.

        Returns
        -------
        str | Path
            The path to the input JSON configuration file.

        """
        return self._input_json_path

    @abstractmethod
    def run_simulation(self):
        """Run the simulation for the given JSON configuration file.

        This method must be implemented by all subclasses to perform
        the actual simulation computation.

        """
        pass

    def save_results(
        self,
        url="http://host.docker.internal:5001/receive",
        max_retries=5,
        delay=2,
    ):
        """Return the results back to the simulation service executor.

        This method sends the simulation results stored in the input JSON file
        back to the backend service via HTTP POST request.

        Parameters
        ----------
        url : str, optional
            The URL of the results server. Default is
            "http://host.docker.internal:5001/receive" which is the
            standard address for local execution via Docker.
        max_retries : int, optional
            The maximum number of retries if the request fails.
            Default is 5.
        delay : int, optional
            The delay in seconds between retries. Default is 2.

        Returns
        -------
        bool
            True if results were successfully sent, False otherwise.

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
                    f"Server returned {response.status_code}",
                )
            except requests.RequestException as exc:
                print(f"Attempt {attempt}: Request failed - {exc}")

            time.sleep(delay)

        print("Max retries reached. Giving up.")
        return False
