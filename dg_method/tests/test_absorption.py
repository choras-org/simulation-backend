import numpy as np
import pytest

from dg_interface.absorption import (
    ALPHA_RANDOM_INCIDENCE_MAX,
    bands_up_to_frequency,
    dg_reflection_coefficient,
    reflection_coefficient_from_random_incidence_alpha,
)

BANDS_APP = [125, 250, 500, 1000, 2000]


def _random_incidence_alpha_numeric(R):
    """Independent check: integrate 1 - |R(theta)|^2 over incidence angle."""
    z = (1 + R) / (1 - R)
    theta = np.linspace(0, np.pi / 2, 200001)
    R_theta = (z * np.cos(theta) - 1) / (z * np.cos(theta) + 1)
    f = (1 - R_theta**2) * np.sin(2 * theta)
    return np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(theta))  # trapezoidal rule


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
def test_reflection_coefficient_reproduces_random_incidence_alpha(alpha):
    R = reflection_coefficient_from_random_incidence_alpha(alpha)
    assert 0 < R < 1
    assert _random_incidence_alpha_numeric(R) == pytest.approx(alpha, abs=1e-4)


def test_reflection_coefficient_differs_from_normal_incidence_formula():
    # sqrt(1 - alpha) would treat alpha as normal incidence (too absorptive)
    assert reflection_coefficient_from_random_incidence_alpha(0.1) > np.sqrt(0.9)


def test_reflection_coefficient_limits():
    assert reflection_coefficient_from_random_incidence_alpha(0.0) == 1.0
    R_max = reflection_coefficient_from_random_incidence_alpha(
        ALPHA_RANDOM_INCIDENCE_MAX
    )
    assert reflection_coefficient_from_random_incidence_alpha(0.99) == R_max


@pytest.mark.parametrize(
    ("freq_upper_limit", "expected"),
    [
        (50, [125]),
        (150, [125]),
        (200, [125, 250]),
        (500, [125, 250, 500]),
        (800, [125, 250, 500, 1000]),
        (3000, BANDS_APP),
    ],
)
def test_bands_up_to_frequency(freq_upper_limit, expected):
    n = bands_up_to_frequency(BANDS_APP, freq_upper_limit)
    assert BANDS_APP[:n] == expected


def test_dg_reflection_coefficient_averages_selected_bands():
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    R, alpha_mean, bands = dg_reflection_coefficient(alphas, BANDS_APP, 200)
    assert bands == [125, 250]
    assert alpha_mean == pytest.approx(0.2)
    assert R == pytest.approx(reflection_coefficient_from_random_incidence_alpha(0.2))


def test_dg_reflection_coefficient_rejects_band_mismatch():
    with pytest.raises(ValueError):
        dg_reflection_coefficient([0.1, 0.2], BANDS_APP, 200)
