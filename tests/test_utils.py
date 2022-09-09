import numpy.testing as npt
import numpy as np
from matplotlib import pyplot as plt
import pytest

from utils import celest_to_pa

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

vernal_eq = Time('2021-03-20T00:00:00', format='isot', scale='utc')

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

    npt.assert_almost_equal(celest_to_pa(*test), expected, decimal=1)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([60.0, -10.0, vernal_eq, EarthLocation(lon=0.0, lat=0.0, height=0.0)], 95.8),
        ([90.0, -10.0, vernal_eq, EarthLocation(lon=0.0, lat=0.0, height=0.0)], 90.0),
        ([30.0, -10.0, vernal_eq, EarthLocation(lon=0.0, lat=0.0, height=0.0)], 107.0)
    ])

def test_celest_to_pa_EQ(test, expected):

    npt.assert_almost_equal(celest_to_pa(*test), expected, decimal=1)

'''
def test_pa_plot():

    from utils import pa_plot
    pa_plot([0.0, 0.0, 0.0, 0.0, -5.0, -2.5, 2.5, 5.0], 
            [-5.0, -2.5, 2.5, 5.0, 0.0, 0.0, 0.0, 0.0], 
    vernal_eq, EarthLocation(lon=0.0, lat=0.0, height=0.0))
    plt.show()
'''