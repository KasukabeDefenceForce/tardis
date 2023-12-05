from pathlib import Path
import pandas as pd
import pytest
import re
import os
import numpy as np


class RegressionData:
    def __init__(self, request) -> None:
        self.request = request
        tardis_ref_path = request.config.getoption("--tardis-refdata")
        self.tardis_ref_path = Path(
            os.path.expandvars(os.path.expanduser(tardis_ref_path))
        )
        self.enable_generate_reference = request.config.getoption(
            "--generate-reference"
        )

    @property
    def module_name(self):
        return self.request.node.module.__name__

    @property
    def test_name(self):
        return self.request.node.name

    @property
    def regression_data_fname_prefix(self):
        double_under = re.compile(r"[:\[\]{}]")
        no_space = re.compile(r'[,"\']')  # quotes and commas

        name = double_under.sub("__", self.test_name)
        name = no_space.sub("", name)
        return name

    @property
    def relative_regression_data_dir(self):
        return Path(self.module_name.replace(".", "/"))

    def check_data(self, data):
        full_fname_prefix = (
                self.tardis_ref_path
                / self.regression_data_fname_prefix
        )
        if self.enable_generate_reference:
            if hasattr(data, "to_hdf"):
                data.to_hdf(
                    full_fname_prefix.with_suffix(".h5"),
                    key=self.regression_data_fname_prefix,
                )
            # Numpy
            elif isinstance(data, np.ndarray):
                np.save(full_fname_prefix.with_suffix(".npy"), data)
            elif isinstance(data, str):
                np.savetxt(
                    full_fname_prefix.with_suffix(".txt"),
                    [data],
                    fmt="%s",
                    encoding="utf-8"
                )
            elif np.issubdtype(type(data), np.floating):
                np.savetxt(
                    full_fname_prefix.with_suffix(".txt"),
                    [data],
                )
            pytest.skip("Skipping test to generate reference data")
        else:
            if hasattr(data, "to_hdf"):
                ref_data = pd.read_hdf(
                    full_fname_prefix.with_suffix(".h5"),
                    key=self.regression_data_fname_prefix,
                )
            elif isinstance(data, np.ndarray):
                ref_data = np.load(full_fname_prefix.with_suffix(".npy"))
            elif isinstance(data, str):
                ref_data = np.loadtxt(
                    full_fname_prefix.with_suffix(".txt"),
                    dtype='str',
                    encoding="utf-8"
                )
            elif np.issubdtype(type(data), np.floating):
                ref_data = np.loadtxt(
                    full_fname_prefix.with_suffix(".txt"),
                )
            return ref_data


@pytest.fixture(scope="function")
def regression_data(request):
    return RegressionData(request)
