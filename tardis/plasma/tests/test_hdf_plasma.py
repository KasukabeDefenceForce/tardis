import pandas as pd
import pytest
import pandas.testing as pdt
import numpy.testing as npt
###
# saving and loading of plasma properties in the HDF file
###


plasma_properties_list = [
    "number_density",
    "beta_rad",
    "general_level_boltzmann_factor",
    "level_boltzmann_factor",
    "stimulated_emission_factor",
    "t_electrons",
    "wavelength_cm",
    "lines_lower_level_index",
    "ionization_data",
    "density",
    "atomic_mass",
    "level_number_density",
    "lines_upper_level_index",
    "nu",
    "beta_sobolev",
    "transition_probabilities",
    "phi",
    "electron_densities",
    "t_rad",
    "selected_atoms",
    "ion_number_density",
    "partition_function",
    "abundance",
    "g_electron",
    "g",
    "lines",
    "f_lu",
    "tau_sobolevs",
    "j_blues",
    "metastability",
    "w",
    "excitation_energy",
]


@pytest.mark.parametrize("attr", plasma_properties_list)
def test_hdf_plasma(simulation_verysimple, attr, regression_data):
    if hasattr(simulation_verysimple.plasma, attr):
        actual = getattr(simulation_verysimple.plasma, attr)
        if hasattr(actual, "cgs"):
            actual = actual.cgs.value
        print(type(actual))
        print(actual)
        # assert snapshot_np == actual
        assert False

def test_hdf_levels(simulation_verysimple, regression_data):
    actual = getattr(simulation_verysimple.plasma, "levels")
    if hasattr(actual, "cgs"):
        actual = actual.cgs.value
    actual_df = pd.DataFrame(actual)
    expected_result = regression_data.check_data(actual_df)
    pdt.assert_frame_equal(actual_df, expected_result)


scalars_list = ["time_explosion", "link_t_rad_t_electron"]


@pytest.mark.parametrize("attr", scalars_list)
def test_hdf_scalars(simulation_verysimple, attr, regression_data):
    actual = getattr(simulation_verysimple.plasma, attr)
    if hasattr(actual, "cgs"):
        actual = actual.cgs.value
    expected = regression_data.check_data(actual)
    assert expected == actual


def test_hdf_helium_treatment(simulation_verysimple, regression_data):
    actual = getattr(simulation_verysimple.plasma, "helium_treatment")
    expected = regression_data.check_data(actual)
    assert actual == expected


def test_atomic_data_uuid(simulation_verysimple, regression_data):
    actual = getattr(simulation_verysimple.plasma.atomic_data, "uuid1")
    expected = regression_data.check_data(actual)
    assert expected == actual


collection_properties = ["t_rad", "w", "density"]


@pytest.mark.parametrize("attr", collection_properties)
def test_collection(simulation_verysimple, attr, regression_data):
    actual = getattr(simulation_verysimple.plasma, attr)
    if hasattr(actual, "cgs"):
        actual = actual.cgs.value

    expected = regression_data.check_data(actual)
    npt.assert_allclose(actual, expected)
