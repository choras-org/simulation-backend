"""Conversion of CHORAS absorption coefficients to DG boundary conditions.

CHORAS materials carry random-incidence (diffuse-field) absorption
coefficients per octave band. edg-acoustics models walls as locally reacting
with a frequency-independent, real normal-incidence reflection coefficient R
(incoming characteristic = R * outgoing characteristic). The equivalent
normalised wall impedance is z = (1 + R) / (1 - R).

Using R = sqrt(1 - alpha) treats alpha as a normal-incidence value. Because a
real impedance absorbs more at oblique incidence, that overestimates the
diffuse-field absorption of the DG walls (alpha = 0.1 behaves like ~0.18).
Here R is chosen such that the wall's random-incidence absorption equals the
CHORAS value.
"""
import numpy as np

# Normalised impedance at which the random-incidence absorption of a locally
# reacting wall with real impedance is maximal, and that maximum (~0.951).
_Z_AT_MAX_ALPHA = 1.5671


def random_incidence_alpha(z):
    """Random-incidence absorption of a locally reacting wall (Paris formula).

    Parameters
    ----------
    z : float
        Real normalised wall impedance Z / (rho0 * c0), z > 0.
    """
    return 8.0 / z * (1.0 + 1.0 / (1.0 + z) - 2.0 / z * np.log(1.0 + z))


ALPHA_RANDOM_INCIDENCE_MAX = random_incidence_alpha(_Z_AT_MAX_ALPHA)


def reflection_coefficient_from_random_incidence_alpha(alpha):
    """Normal-incidence reflection coefficient R for a random-incidence alpha.

    Solves random_incidence_alpha(z) = alpha on the high-impedance branch
    (z >= 1.567) and returns R = (z - 1) / (z + 1). Values at or above the
    maximum reachable with a real impedance (~0.951) are limited to it.
    """
    alpha = float(alpha)
    if alpha <= 0.0:
        return 1.0
    if alpha >= ALPHA_RANDOM_INCIDENCE_MAX:
        print(
            f"Warning: random-incidence absorption {alpha:.3f} exceeds the "
            f"maximum {ALPHA_RANDOM_INCIDENCE_MAX:.3f} reachable with a real "
            f"impedance; using {ALPHA_RANDOM_INCIDENCE_MAX:.3f}."
        )
        z = _Z_AT_MAX_ALPHA
    else:
        # Bisection in log(z): alpha decreases monotonically with z on this branch
        lo, hi = np.log(_Z_AT_MAX_ALPHA), np.log(1e12)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if random_incidence_alpha(np.exp(mid)) > alpha:
                lo = mid
            else:
                hi = mid
        z = np.exp(0.5 * (lo + hi))
    return float((z - 1.0) / (z + 1.0))


def bands_up_to_frequency(band_frequencies, freq_upper_limit):
    """Number of octave bands up to and including the one containing a frequency.

    A band with centre frequency fc spans fc / sqrt(2) to fc * sqrt(2). It is
    included when its lower edge lies below ``freq_upper_limit``. At least
    one band is always used.
    """
    n = sum(1 for fc in band_frequencies if fc / np.sqrt(2) < freq_upper_limit)
    return max(1, min(n, len(band_frequencies)))


def dg_reflection_coefficient(alphas, band_frequencies, freq_upper_limit):
    """DG reflection coefficient for one material.

    Averages the random-incidence absorption coefficients over the bands up
    to and including the band containing ``freq_upper_limit`` and converts
    the average to the normal-incidence reflection coefficient.

    Returns
    -------
    R : float
        Normal-incidence reflection coefficient.
    alpha_mean : float
        Averaged random-incidence absorption coefficient.
    bands_used : list
        Centre frequencies of the averaged bands.
    """
    if len(alphas) != len(band_frequencies):
        raise ValueError(
            f"Got {len(alphas)} absorption coefficients for "
            f"{len(band_frequencies)} frequency bands {list(band_frequencies)}"
        )
    n = bands_up_to_frequency(band_frequencies, freq_upper_limit)
    alpha_mean = float(np.mean(alphas[:n]))
    R = reflection_coefficient_from_random_incidence_alpha(alpha_mean)
    return R, alpha_mean, list(band_frequencies[:n])
