"""Geometry tests."""
import numpy as np
from ectopylasm import geometry

import pytest


@pytest.mark.parametrize('d', [None, 1])
def test_plane_surface(d):
    """Test geometry.plane_surface."""
    p = (1, 1, 1)
    n = (0, 1, 0)

    x_lim = (-1, 1)
    z_lim = (-1, 1)

    x, y, z = geometry.plane_surface(p, n, x_lim, z_lim, d=d)

    print(x, y, z)


def test_filter_points_cone():
    """Test geometry.filter_points_cone."""
    xyz = np.array([[0, 0, 0.5],   # inside blue, above red
                    [0, 0, 0.7],   # above blue (i.e. above both), within "thickness" from blue
                    [0, 0, 1],     # far above both
                    [0, 0, 0.3],   # inside red
                    [0, 0, 0.1],   # inside red, "below" blue
                    [0, 0, -0.1],  # below both
                    ]).T

    p_filtered = geometry.filter_points_cone(xyz, 0.5, 0.5, 0.2)
    print(p_filtered)
    assert np.all(np.array(p_filtered) == np.array([[0, 0, 0.5]]))
