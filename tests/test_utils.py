import numpy.testing as npt
import numpy as np
from matplotlib import pyplot as plt
import pytest

from utils import celest_to_pa
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

vernal_eq = Time('2021-03-20T10:00:00', scale='ut1')
fall_eq = Time('2021-09-22T10:00:00', scale='ut1')
summer_sol = Time('2021-06-20T10:00:00', scale='ut1')

arcturus = SkyCoord.from_name('Arcturus')
rigel = SkyCoord.from_name('Rigel')
sirius = SkyCoord.from_name('Sirius')
procyon = SkyCoord.from_name('Procyon')

@pytest.mark.parametrize(
    "test, expected",
    [
        ([0.0, 0.0, vernal_eq, EarthLocation(lon=0.0, lat=90.0, height=0.0)], 0.0),
        ([0.0, 45.0, vernal_eq, EarthLocation(lon=0.0, lat=90.0, height=0.0)], 0.0),
        ([0.0, 90.0, vernal_eq, EarthLocation(lon=0.0, lat=90.0, height=0.0)], 0.0)
    ])
def test_celest_to_pa_NP(test, expected):

    npt.assert_approx_equal(celest_to_pa(*test), expected, significant=2)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([arcturus.ra.value, arcturus.dec.value, vernal_eq, EarthLocation.of_site('Keck')], -81.8),
        ([rigel.ra.value, rigel.dec.value, vernal_eq, EarthLocation.of_site('Keck')], 70.8),
        ([sirius.ra.value, sirius.dec.value, vernal_eq, EarthLocation.of_site('Keck')], 65.2),
        ([procyon.ra.value, procyon.dec.value, vernal_eq, EarthLocation.of_site('Keck')], 69.8),

        ([arcturus.ra.value, arcturus.dec.value, fall_eq, EarthLocation.of_site('Keck')], 45.9),
        ([rigel.ra.value, rigel.dec.value, fall_eq, EarthLocation.of_site('Keck')], -69.3),
        ([sirius.ra.value, sirius.dec.value, fall_eq, EarthLocation.of_site('Keck')], -74.4),
        ([procyon.ra.value, procyon.dec.value, fall_eq, EarthLocation.of_site('Keck')], -65.2)
    ])

def test_celest_to_pa_keck(test, expected):
    print('Test observatory:')
    print(' Keck, Lon: {0}, Lat: {1}, Alt: {2}'.format(EarthLocation.of_site('Keck').geodetic.lon.hms,
                                                       EarthLocation.of_site('Keck').geodetic.lat.hms,
                                                       EarthLocation.of_site('Keck').geodetic.height))
    print('Test stars:')
    print(' Arcturus, ra: {0}, dec: {1}'.format(arcturus.ra.hms, arcturus.dec.dms))
    print(' Rigel, ra: {0}, dec: {1}'.format(rigel.ra.hms, rigel.dec.dms))
    print(' Sirius, ra: {0}, dec: {1}'.format(sirius.ra.hms, sirius.dec.dms))
    print(' Procyon, ra: {0}, dec: {1}'.format(procyon.ra.hms, procyon.dec.dms))

    npt.assert_approx_equal(celest_to_pa(*test, verbose=False), expected, significant=3)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([arcturus.ra.value, arcturus.dec.value, vernal_eq, EarthLocation.of_site('Palomar')], -57.4),
        ([rigel.ra.value, rigel.dec.value, vernal_eq, EarthLocation.of_site('Palomar')], 57.5),
        ([sirius.ra.value, sirius.dec.value, vernal_eq, EarthLocation.of_site('Palomar')], 55.3),

        ([arcturus.ra.value, arcturus.dec.value, fall_eq, EarthLocation.of_site('Palomar')], 39.5),
        ([rigel.ra.value, rigel.dec.value, fall_eq, EarthLocation.of_site('Palomar')], -56.8),
        ([sirius.ra.value, sirius.dec.value, fall_eq, EarthLocation.of_site('Palomar')], -60.0)
    ])

def test_celest_to_pa_palomar(test, expected):
    
    npt.assert_approx_equal(celest_to_pa(*test, verbose=True), expected, significant=3)