import numpy.testing as npt
import numpy as np
from matplotlib import pyplot as plt
import pytest

from utils import celest_to_pa

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

vernal_eq = Time('2021-03-20T00:00:00', format='isot', scale='utc')
fall_eq = Time('2021-09-22T00:00:00', format='isot', scale='utc')
summer_sol = Time('2021-06-20T00:00:00', format='isot', scale='utc')

arcturus = SkyCoord.from_name('Arcturus')
rigel = SkyCoord.from_name('Rigel')
sirius = SkyCoord.from_name('Sirius')

@pytest.mark.parametrize(
    "test, expected",
    [
        ([0.0, 0.0, vernal_eq, EarthLocation(lon=0.0, lat=90.0, height=0.0)], 0.0),
        ([0.0, 45.0, vernal_eq, EarthLocation(lon=0.0, lat=90.0, height=0.0)], 0.0),
        ([45.0, 45.0, vernal_eq, EarthLocation(lon=0.0, lat=90.0, height=0.0)], 0.0)
    ])
def test_celest_to_pa_NP(test, expected):

    npt.assert_almost_equal(celest_to_pa(*test), expected, decimal=1)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([0.0, 0.0, vernal_eq, EarthLocation(lon=0.0, lat=-90.0, height=0.0)], 180.0),
        ([0.0, 45.0, vernal_eq, EarthLocation(lon=0.0, lat=-90.0, height=0.0)], 180.0),
        ([45.0, 45.0, vernal_eq, EarthLocation(lon=0.0, lat=-90.0, height=0.0)], 180.0)
    ])
def test_celest_to_pa_SP(test, expected):

    npt.assert_almost_equal(celest_to_pa(*test), expected, decimal=2)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([arcturus.ra.value, arcturus.dec.value, vernal_eq, EarthLocation.of_site('Keck')], -81.9),
        ([rigel.ra.value, rigel.dec.value, vernal_eq, EarthLocation.of_site('Keck')], 65.5),
        ([sirius.ra.value, sirius.dec.value, vernal_eq, EarthLocation.of_site('Keck')], 70.1),

        ([arcturus.ra.value, arcturus.dec.value, fall_eq, EarthLocation.of_site('Keck')], 45.1),
        ([rigel.ra.value, rigel.dec.value, fall_eq, EarthLocation.of_site('Keck')], -70.0),
        ([sirius.ra.value, sirius.dec.value, fall_eq, EarthLocation.of_site('Keck')], -69.6)
    ])

def test_celest_to_pa_keck(test, expected):

    npt.assert_almost_equal(celest_to_pa(*test, verbose=True), expected, decimal=2)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([arcturus.ra.value, arcturus.dec.value, vernal_eq, EarthLocation.of_site('Palomar')], -56.9),
        ([rigel.ra.value, rigel.dec.value, vernal_eq, EarthLocation.of_site('Palomar')], 57.5),
        ([sirius.ra.value, sirius.dec.value, vernal_eq, EarthLocation.of_site('Palomar')], 55.5),

        ([arcturus.ra.value, arcturus.dec.value, fall_eq, EarthLocation.of_site('Palomar')], 38.8),
        ([rigel.ra.value, rigel.dec.value, fall_eq, EarthLocation.of_site('Palomar')], -56.7),
        ([sirius.ra.value, sirius.dec.value, fall_eq, EarthLocation.of_site('Palomar')], -61.7)
    ])

def test_celest_to_pa_palomar(test, expected):

    npt.assert_almost_equal(celest_to_pa(*test, verbose=True), expected, decimal=2)