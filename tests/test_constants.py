"""Tests of ``ppf_c5_runaways.constants``."""

import math

import pytest
import scipy.constants as scipy_constants

from ppf_c5_runaways import constants


def test_si_defined_constants_are_exact():
    assert constants.c == 299_792_458.0
    assert constants.e == 1.602176634e-19


@pytest.mark.parametrize(
    ("ours", "scipy_name"),
    [
        (constants.c, "speed of light in vacuum"),
        (constants.e, "elementary charge"),
        (constants.m_e, "electron mass"),
        (constants.eps_0, "vacuum electric permittivity"),
        (constants.mu_0, "vacuum mag. permeability"),
    ],
)
def test_constants_agree_with_scipy(ours, scipy_name):
    # The tolerance 1e-8 is deliberately loose:
    # the project accepts scipy>=1.14, which still ships CODATA 2018, and the two
    # editions differ by up to 1.4e-9 (electron mass). This test catches a wrong
    # digit, not the choice of edition; the tests below pin the digits more tightly.
    reference = scipy_constants.physical_constants[scipy_name][0]

    assert ours == pytest.approx(reference, rel=1e-8)


def test_light_speed_relation():
    # Maxwell: c^2 * eps_0 * mu_0 = 1. Tolerance 1e-9: about 6 times the relative
    # standard uncertainty of eps_0 and mu_0 (1.6e-10), much larger than the
    # rounding of their 11 printed digits (~1e-11).
    product = constants.c**2 * constants.eps_0 * constants.mu_0

    assert product == pytest.approx(1.0, rel=1e-9)


def test_electron_rest_energy_in_mev():
    # Independent truth: the CODATA 2022 value of m_e c^2 in MeV, a quantity not
    # stored in the module (it combines m_e, c and e). Tolerance 1e-9: about 3
    # times its relative uncertainty (3.1e-10).
    rest_energy_mev = constants.m_e * constants.c**2 / constants.e / 1.0e6

    assert rest_energy_mev == pytest.approx(0.51099895069, rel=1e-9)


def test_classical_electron_radius():
    # Independent truth: the CODATA 2022 classical electron radius, 2.8179403205e-15
    # m. It combines e, eps_0, m_e and c, so an error in eps_0 shows up here.
    # Tolerance 1e-9: about 2 times its relative uncertainty (4.6e-10).
    radius = constants.e**2 / (
        4.0 * math.pi * constants.eps_0 * constants.m_e * constants.c**2
    )

    assert radius == pytest.approx(2.8179403205e-15, rel=1e-9)


def test_elementary_charge_is_positive():
    # The sign of a charge is carried by the caller (q = -e for an electron): a
    # negative ``e`` here would silently reverse every gyration.
    assert constants.e > 0.0
