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
    xyz = np.array([[0, 0, 0.5],   # inside blue, above red
                    [0, 0, 0.7],   # above blue (i.e. above both), within "thickness" from blue
                    [0, 0, 1],     # far above both
                    [0, 0, 0.3],   # inside red
                    [0, 0, 0.1],   # inside red, "below" blue
                    [0, 0, -0.1],  # below both
                    ]).T

    cone = geometry.Cone(0.5, 0.5)
    p_filtered = geometry.filter_points_cone(xyz, cone, 0.2)
    assert np.all(np.array(p_filtered) == np.array([[0, 0, 0.5]]))
