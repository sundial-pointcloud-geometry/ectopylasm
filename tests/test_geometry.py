"""Geometry tests."""
import numpy as np
from ectopylasm import geometry

import pytest


@pytest.mark.parametrize('d', [None, -1])
def test_plane_surface(d):
    """Test geometry.plane_surface."""
    p = (1, 1, 1)
    n = (0, 1, 0)

    x_lim = (-1, 1)
    z_lim = (-1, 1)

    x, y, z = geometry.plane_surface(p, n, x_lim, z_lim, d=d)

    assert np.all(x == np.array([[-1, 1], [-1, 1]]))
    assert np.all(y == np.array([[1., 1.], [1., 1.]]))
    assert np.all(z == np.array([[-1, -1], [1, 1]]))


def test_filter_points_plane():
    """Test geometry.filter_points_plane."""
    xyz = np.array([[0, 0.61, 0],   # just above
                    [0, 0.39, 0],   # just below
                    [0, 0.5, 0],    # right in the middle
                    ]).T

    p_filtered = geometry.filter_points_plane(xyz, (0.5, 0.5, 0.5), (0, 1, 0), 0.2)
    assert np.all(np.array(p_filtered) == np.array([[0, 0.5, 0]]))


def test_filter_points_cone():
    """Test geometry.filter_points_cone."""
    xyz = np.array([[0, 0, 0.5],     # inside blue, above red ("case 2")
                    [0, 0.01, 0.5 + np.sqrt(0.2**2 / 2)],  # slightly besides blue ("case 3")
                    [0, 0.2, 0.3],   # somewhere between the directrices ("case 4")
                    [0, 0, 0.7],     # above blue (i.e. above both), within "thickness" from blue ("case 1")
                    [0, 0, 1],       # far above both ("case 1")
                    [0, 0, 0.3],     # inside red ("case 5")
                    [0, 0, 0.1],     # inside red, "below" blue ("case 5")
                    [0, 0, -0.7],    # below both ("case 1")
                    ]).T

    cone = geometry.Cone(0.5, 0.5)
    p_filtered = geometry.filter_points_cone(xyz, cone, 0.2)
    assert np.all(np.array(p_filtered) == np.array([[0, 0, 0.5], [0, 0.2, 0.3]]))


def test_cone_surface():
    """Test geometry.cone_surface."""
    cone = geometry.Cone(0.5, 0.5)

    xyz = np.array(geometry.cone_surface(cone, n_steps=3))

    # using buffer to preserve exact, non-rounded floating point numbers
    prebaked = np.frombuffer(b'\x00\x00\x00\x00\x00\x00\xe0?\x01\x00\x00\x00\x00\x00\xd0?\x07\\\x143&\xa6\xa1<\x00\x00\x00\x00\x00\x00\xe0\xbf\xfe\xff\xff\xff\xff\xff\xcf\xbf\x07\\\x143&\xa6\xa1<\x00\x00\x00\x00\x00\x00\xe0?\x01\x00\x00\x00\x00\x00\xd0?\x07\\\x143&\xa6\xa1<\x00\x00\x00\x00\x00\x00\x00\x00\x07\\\x143&\xa6\x91<\x07\\\x143&\xa6\xa1<\x07\\\x143&\xa6\x91<\n\x8a\x9eL9y\x9a<\x07\\\x143&\xa6\xa1<\x07\\\x143&\xa6\xa1\xbc\xbf\x8f\xed\xb7v\x19\x1f9\x07\\\x143&\xa6\xa1<\x07\\\x143&\xa6\xa1\xbc\xfe\xff\xff\xff\xff\xff\xcf?\x00\x00\x00\x00\x00\x00\xe0?\x06\\\x143&\xa6\xa1<\x01\x00\x00\x00\x00\x00\xd0?\x00\x00\x00\x00\x00\x00\xe0?\x06\\\x143&\xa6\xa1\xbc\xfe\xff\xff\xff\xff\xff\xcf?\x00\x00\x00\x00\x00\x00\xe0?').reshape(3, 3, 3)

    assert np.all(xyz == prebaked)


def test_thick_cone_base_positions():
    """Test thick_cone_base_positions."""
    cone = geometry.Cone(0.5, 0.5)

    thickness = 0.2

    theta = cone.opening_angle()
    base_pos_distance = thickness / np.sin(theta)

    b1, b2 = geometry.thick_cone_base_positions(cone, thickness)
    assert np.allclose(np.array(b1), -np.array((0, 0, base_pos_distance / 2)))
    assert np.allclose(np.array(b2), np.array((0, 0, base_pos_distance / 2)))


@pytest.mark.parametrize('point,expected_distance,expected_flag', [
    ([0, 0, 0], -0.5 / np.sqrt(2), geometry.ConeRegion.perpendicular),     # at the base center
    ([0, 0, 0.5], 0, geometry.ConeRegion.perpendicular),                  # at the apex
    ([0.1, 0, 0.4], 0, geometry.ConeRegion.perpendicular),              # on a generatrix
    ([0, 0, 0.6], 0.1, geometry.ConeRegion.above_apex),               # above the apex
    ([0, -2, -0.5 - 1], np.sqrt(2 * (2 * 0.5 + 1)**2) - np.sqrt(2 * (0.5)**2), geometry.ConeRegion.below_directrix),  # below the directrix, but along the generatrix direction
])
def test_point_distance_to_cone(point, expected_distance, expected_flag):
    """Test point_distance_to_cone."""
    cone = geometry.Cone(0.5, 0.5)
    distance, flag, _ = geometry.point_distance_to_cone(point, cone)
    assert np.isclose(distance, expected_distance)
    assert flag == expected_flag
