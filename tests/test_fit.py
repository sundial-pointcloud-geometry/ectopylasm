"""Fitting tests."""
import numpy as np
from ectopylasm import fit, geometry


def test_fit_plane():
    """Test fit_plane."""
    np.random.seed(1)
    xyz = np.array((np.random.random(1000) + np.linspace(-100, 0, 1000), np.random.normal(0, 0.01, 1000) + np.linspace(-3, 3, 1000), np.random.random(1000)))

    fit_result = fit.fit_plane(xyz)
    assert np.isclose(fit_result.params['a'], -5.988609e-02)
    assert np.isclose(fit_result.params['b'], 9.981877e-01)
    assert np.isclose(fit_result.params['c'], -5.909291e-03)
    assert np.isclose(fit_result.params['d'], -2.961908e+00)


def test_fit_cone_initial_guess():
    """Test fit_cone with an initial guess parameter."""
    cone = geometry.Cone(0.5, 0.5, rot_x=3., rot_y=1., base_pos=geometry.Point(8, 10, -48))

    np.random.seed(1)
    n_steps = 10
    xyz = np.array(geometry.cone_surface(cone, n_steps=n_steps))
    xyz = xyz.reshape(3, n_steps * n_steps)

    initial_guess = geometry.Cone(0.51, 0.51, rot_x=3.01, rot_y=0.99, base_pos=geometry.Point(7.9, 10., -48.1))

    fit_result = fit.fit_cone(xyz, initial_guess_cone=initial_guess)

    assert np.isclose(fit_result['x'][0], 0.5069517)
    assert np.isclose(fit_result['x'][1], 0.50695155)
    assert np.isclose(fit_result['x'][2], 3.00000021)
    assert np.isclose(fit_result['x'][3], 1.00000017)
    assert np.isclose(fit_result['x'][4], 7.99420898)
    assert np.isclose(fit_result['x'][5], 10.00098104)
    assert np.isclose(fit_result['x'][6], -47.99628163)
