"""Fitting tests."""
import numpy as np
from ectopylasm import geometry

import pytest


@pytest.fixture
def xyz_plane():
    """Fixture to generate random set of plane points."""
    np.random.seed(1)
    xyz = np.array((np.random.random(1000) + np.linspace(-100, 0, 1000), np.random.normal(0, 0.01, 1000) + np.linspace(-3, 3, 1000), np.random.random(1000)))
    return xyz


@pytest.fixture
def plane_fit_result(xyz_plane):
    """Fixture to fit points to plane."""
    return geometry.fit_plane(xyz_plane)


def test_fit_plane(plane_fit_result):
    """Test fit_plane."""
    fit_result = plane_fit_result
    assert np.isclose(fit_result.params['a'], -5.988609e-02)
    assert np.isclose(fit_result.params['b'], 9.981877e-01)
    assert np.isclose(fit_result.params['c'], -5.909291e-03)
    assert np.isclose(fit_result.params['d'], -2.961908e+00)


def test_plane_from_fit_result_constructor(plane_fit_result):
    """Test the Plane constructors that use fitting to construct the Plane."""
    plane = geometry.Plane.from_fit_result(plane_fit_result)
    assert np.isclose(plane.a, -5.988609e-02)
    assert np.isclose(plane.b, 9.981877e-01)
    assert np.isclose(plane.c, -5.909291e-03)
    assert np.isclose(plane.d, -2.961908e+00)


def test_plane_from_points_constructor(xyz_plane):
    """Test the Plane constructors that use fitting to construct the Plane."""
    plane = geometry.Plane.from_points(xyz_plane)
    assert np.isclose(plane.a, -5.988609e-02)
    assert np.isclose(plane.b, 9.981877e-01)
    assert np.isclose(plane.c, -5.909291e-03)
    assert np.isclose(plane.d, -2.961908e+00)


@pytest.fixture
def xyz_cone():
    """Fixture to generate random set of cone points."""
    cone = geometry.Cone(0.5, 0.5, rot_x=3., rot_y=1., base_pos=geometry.Point(8, 10, -48))

    np.random.seed(1)
    n_steps = 10
    xyz = np.array(geometry.cone_surface(cone, n_steps=n_steps))
    xyz = xyz.reshape(3, n_steps * n_steps)

    return xyz


@pytest.fixture
def initial_cone():
    """Fixture to generate initial cone guess."""
    return geometry.Cone(0.51, 0.51, rot_x=3.01, rot_y=0.99, base_pos=geometry.Point(7.9, 10., -48.1))


@pytest.fixture
def cone_fit_result(xyz_cone, initial_cone):
    """Fixture to fit points to cone."""
    return geometry.fit_cone(xyz_cone, initial_guess_cone=initial_cone)


def test_fit_cone_initial_guess(cone_fit_result):
    """Test fit_cone with an initial guess parameter."""
    fit_result = cone_fit_result

    assert np.isclose(fit_result['x'][0], 0.5069517)
    assert np.isclose(fit_result['x'][1], 0.50695155)
    assert np.isclose(fit_result['x'][2], 3.00000021)
    assert np.isclose(fit_result['x'][3], 1.00000017)
    assert np.isclose(fit_result['x'][4], 7.99420898)
    assert np.isclose(fit_result['x'][5], 10.00098104)
    assert np.isclose(fit_result['x'][6], -47.99628163)


def test_cone_from_fit_result_constructors(cone_fit_result):
    """Test the Cone constructors that use fitting to construct the Cone."""
    cone = geometry.Cone.from_fit_result(cone_fit_result)

    assert np.isclose(cone.height, 0.5069517)
    assert np.isclose(cone.radius, 0.50695155)
    assert np.isclose(cone.rot_x, 3.00000021)
    assert np.isclose(cone.rot_y, 1.00000017)
    assert np.isclose(cone.base_pos.x, 7.99420898)
    assert np.isclose(cone.base_pos.y, 10.00098104)
    assert np.isclose(cone.base_pos.z, -47.99628163)
