import json
import os
import pandas as pd
from pathlib import Path

from acousticDE.FiniteVolumeMethod.FVM import run_fvm_sim
from acousticDE.FiniteVolumeMethod.CreateMeshFVM import generate_mesh

from .definition import SimulationMethod


class DEMethod(SimulationMethod):
    """Interface class to run the DE method.

    The class implements method to run the calculations for the
    finite volume method. All required configuration parameters are expected
    to be provided in the input JSON file passed during initialization.

    """

    def __init__(self, input_json_path: str | Path | None = None):
        """Initialize the DE method interface for the given JSON file."""
        super().__init__(input_json_path)

    def run_simulation(self, json_file_path: str | Path | None = None) -> None:
        """Run the simulation.

        Parameters
        ----------
        json_file_path : str | Path | None, optional
            Path to the JSON file. If not provided, uses the path from initialization.
        """
        if json_file_path is None:
            json_file_path = self.input_json_path
        self._de_method(json_file_path)  # type: ignore

    def _de_method(self, json_file_path: str | Path) -> None:
        """
        Run DE simulation for acoustic wave propagation.

        Args:
            json_file_path: Path to the JSON configuration file
        """
        result_container = {}
        with open(json_file_path, "r") as json_file:
            result_container = json.load(json_file)

        de_data = {}

        ## Prepare json input data for run_fvm_sim from the json input data

        # Source coordinates
        de_data["coord_source"] = [
            result_container["results"][0]["sourceX"],
            result_container["results"][0]["sourceY"],
            result_container["results"][0]["sourceZ"],
        ]

        # Receiver coordinates
        de_data["coord_rec"] = [
            result_container["results"][0]["responses"][0]["x"],
            result_container["results"][0]["responses"][0]["y"],
            result_container["results"][0]["responses"][0]["z"],
        ]

        # Frequency bands
        freqs = result_container["results"][0]["frequencies"]
        de_data["fc_low"] = freqs[0]
        de_data["fc_high"] = freqs[-1]

        # Absorption coefficients (create csv necessary for run_fvm_sim)
        csv_path = os.path.join(
            os.path.dirname(str(json_file_path)),
            "absorption_coefficients.csv"
        )

        surface_names = []
        for key, value in result_container["absorption_coefficients"].items():
            surface_names.append(key)

        column_names = ["Material"] + [f"{int(fc)}Hz" for fc in freqs]

        # Convert JSON dict to dataframe
        records = []
        for material, coeffs in result_container[
            "absorption_coefficients"
        ].items():
            coeff_list = [float(c.strip()) for c in coeffs.split(",")]
            records.append([material] + coeff_list)

        df = pd.DataFrame(records, columns=column_names)

        # Save dataframe to CSV
        df.to_csv(csv_path, index=False)

        # Octave bands? (Ask Ilaria)
        de_data["num_octave"] = 1

        # Time step
        de_data["dt"] = 1 / 20000

        # Air absorption coefficient
        de_data["m_atm"] = 0

        # Absorption condition
        # (options Sabine (th=1), Eyring (th=2) and modified by Xiang (th=3))
        de_data["th"] = 3

        de_data["sim_len_type"] = result_container[
            "simulationSettings"
        ]["sim_len_type"]
        de_data["de_ir_length"] = result_container[
            "simulationSettings"
        ]["de_ir_length"]
        de_data["edt"] = result_container["simulationSettings"]["edt"]

        # Append the result container with the data necessary for DE
        result_container.update(de_data)

        # Write the data to the json file
        with open(json_file_path, "w") as json_output:
            json_output.write(json.dumps(result_container, indent=4))

        geo_file_path = result_container["geo_path"]
        msh_file_path = result_container["msh_path"]
        generate_mesh(
            geo_file_path,
            msh_file_path,
            result_container["simulationSettings"]["de_lc"],
        )

        # Run the simulation and obtain the results
        results = run_fvm_sim(
            result_container["msh_path"], json_file_path, csv_path
        )

        ## Write the results to the correct locations in the result container
        result_container["results"][0]["responses"][0]["parameters"][
            "edt"
        ] = results["edt_band"].tolist()
        result_container["results"][0]["responses"][0]["parameters"][
            "t20"
        ] = results["t20_band"].tolist()
        result_container["results"][0]["responses"][0]["parameters"][
            "t30"
        ] = results["t30_band"].tolist()
        result_container["results"][0]["responses"][0]["parameters"][
            "c80"
        ] = results["c80_band"].tolist()
        result_container["results"][0]["responses"][0]["parameters"][
            "d50"
        ] = results["d50_band"].tolist()
        result_container["results"][0]["responses"][0]["parameters"][
            "ts"
        ] = results["ts_band"].tolist()
        result_container["results"][0]["responses"][0]["parameters"][
            "spl_t0_freq"
        ] = results["spl_r_t0_band"]

        df = pd.DataFrame()
        for index, (edc_detail, pressure_detail) in enumerate(
            zip(results["spl_r_off_band"], results["p_rec_off_deriv_band"])
        ):
            result_container["results"][0]["responses"][0][
                "receiverResults"
            ].append(
                {
                    "data": edc_detail.tolist(),
                    "t": (results["t_off"] - results["t_off"][0]).tolist(),
                    "frequency": result_container["results"][0][
                        "frequencies"
                    ][index],
                    "type": "edc",
                }
            )
            if "t" not in df.columns:
                df["t"] = (
                    results["t_off"] - results["t_off"][0]
                ).tolist()
            df[
                str(result_container["results"][0]["frequencies"][index])
                + "Hz"
            ] = pressure_detail.tolist()

        result_container["results"][0]["percentage"] = 100

        # write results to the json
        with open(json_file_path, "w") as new_result_json:
            new_result_json.write(json.dumps(result_container, indent=4))

        # export to .csv
        with open(
            str(json_file_path).replace(".json", "_pressure.csv"),
            "w",
            newline="",
        ) as pressure_result_csv:
            df.to_csv(pressure_result_csv, index=False)
