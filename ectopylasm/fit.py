"""Fitting of point cloud data to geometrical shapes."""
import numpy as np
import symfit as sf


def fit_plane(xyz):
    """
    Fit a plane to the point coordinates in xyz.

    Dev note: An alternative implementation is possible that omits the `f`
    variable, and thus has one fewer degree of freedom. This means the fit is
    easier and maybe more precise. This could be tested. The notebook
    req4.1_fit_plane.ipynb in the explore repository
    (https://github.com/sundial-pointcloud-geometry/explore) has some notes on
    this. The problem with those models where f is just zero and the named
    symfit model is created for one of x, y or z instead is that you have to
    divide by one of the a, b or c parameters respectively. If one of these
    turns out to be zero, symfit will not find a fit. A solution would be
    to actually create three models and try another if one of them fails to
    converge.
    """
    a, b, c, d = sf.parameters('a, b, c, d')
    x, y, z, f = sf.variables('x, y, z, f')
    plane_model = {f: x * a + y * b + z * c - d}

    plane_fit = sf.Fit(plane_model, x=xyz[0], y=xyz[1], z=xyz[2],
                       f=np.zeros_like(xyz[0]),
                       constraints=[sf.Equality(a**2 + b**2 + c**2, 1)])

    plane_fit_result = plane_fit.execute()

    return plane_fit_result
