"""Visualization of point cloud data and geometrical shapes."""

import logging

import numpy as np
import ipyvolume as ipv
import pptk

from ectopylasm import geometry


LOGGER = logging.getLogger('ectopylasm.visualize')
LOGGER.setLevel(logging.INFO)


def random_sample(xyz, total, sample_frac):
    """
    Get a random sample from a point cloud.

    xyz: array of shape (3, N) representing N points in 3D space
    total: number of points in xyz
    sample_frac: fraction of the total set that you want to return
    """
    sample_size = int(sample_frac * total)
    sample = np.random.choice(total, sample_size, replace=False)
    LOGGER.debug("sample size: %i out of total %i", sample_size, total)
    return dict(x=xyz['x'][sample], y=xyz['y'][sample], z=xyz['z'][sample])


def ipv_plot_plydata(plydata, sample_frac=1, marker='circle_2d', **kwargs):
    """Plot vertices in a plydata object using ipyvolume."""
    if sample_frac < 1:
        xyz = random_sample(plydata['vertex'], plydata['vertex'].count, sample_frac)
    else:
        xyz = dict(x=plydata['vertex']['x'], y=plydata['vertex']['y'], z=plydata['vertex']['z'])
    fig = ipv.scatter(**xyz, marker=marker, **kwargs)
    return fig


def pptk_plot_plydata(plydata, **kwargs):
    """Plot vertices in a plydata object using pptk."""
    pptk.viewer(np.array([plydata['vertex']['x'], plydata['vertex']['y'],
                          plydata['vertex']['z']]).T, **kwargs)


def ipv_plot_df(points_df, sample_frac=1, marker='circle_2d', size=0.2, **kwargs):
    """Plot vertices in a dataframe using ipyvolume."""
    if sample_frac < 1:
        xyz = random_sample(points_df, len(points_df), sample_frac)
    else:
        xyz = dict(x=points_df['x'].values, y=points_df['y'].values, z=points_df['z'].values)
    fig = ipv.scatter(**xyz, marker=marker, size=size, **kwargs)
    return fig


def pptk_plot_df(points_df, **kwargs):
    """Plot vertices in a dataframe using pptk."""
    pptk.viewer(np.array([points_df['x'], points_df['y'], points_df['z']]).T, **kwargs)


def plot_plane(plane: geometry.Plane, x_lim=None, y_lim=None, z_lim=None,
               limit_all=True, **kwargs):
    """
    Draw a plane.

    The limited coordinates are called x and z, corresponding to the first and
    third components of `p` and `n`. The final y coordinate is calculated
    based on the equation for a plane.

    plane: a Plane object
    x_lim [optional]: iterable of the two extrema in the x direction
    y_lim [optional]: same as x, but for y
    z_lim [optional]: same as x, but for z
    limit_all [optional]: make sure that the plane surface plot is bound within
                          all given limits
    """
    if x_lim is None:
        x_lim = ipv.pylab.gcf().xlim
    if y_lim is None:
        y_lim = ipv.pylab.gcf().ylim
    if z_lim is None:
        z_lim = ipv.pylab.gcf().zlim
    x, y, z = geometry.plane_surface(plane, x_lim=x_lim, y_lim=y_lim,
                                     z_lim=z_lim)
    fig = ipv.plot_surface(x, y, z, **kwargs)
    if limit_all:
        if np.any(x < x_lim[0]) or np.any(x > x_lim[1]):
            ipv.pylab.xlim(*x_lim)
        if np.any(y < y_lim[0]) or np.any(y > y_lim[1]):
            ipv.pylab.ylim(*y_lim)
        if np.any(z < z_lim[0]) or np.any(z > z_lim[1]):
            ipv.pylab.zlim(*z_lim)
    return fig


def plot_thick_plane(plane: geometry.Plane, thickness=0, **kwargs):
    """
    Draw two co-planar planes, separated by a distance `thickness`.

    plane: a central Plane object
    thickness: the distance between the two co-planar planes
    x_lim [optional]: iterable of the two extrema in the x direction
    z_lim [optional]: same as x, but for z
    d [optional]: if d is known (in-product of p and n), then this can be
                  supplied directly; p is disregarded in this case.
    """
    if thickness <= 0:
        fig = plot_plane(plane, **kwargs)
    else:
        plane_1, plane_2 = geometry.thick_plane_planes(plane, thickness)

        plot_plane(plane_1, **kwargs)
        fig = plot_plane(plane_2, **kwargs)
    return fig


def plot_plane_fit(fit_result, **kwargs):
    """Plot the plane resulting from a plane fit to a point set."""
    p_fit = fit_result.params
    plane = geometry.Plane(p_fit['a'], p_fit['b'], p_fit['c'], p_fit['d'])
    fig = plot_plane(plane, **kwargs)
    return fig


def plot_cone(cone: geometry.Cone, n_steps=20, **kwargs):
    """
    Draw a cone surface.

    cone: a Cone object
    n_steps: number of steps in the parametric range used for drawing (more gives a
             smoother surface, but may render more slowly)
    """
    fig = ipv.plot_surface(*geometry.cone_surface(cone, n_steps=n_steps), **kwargs)
    return fig


def plot_thick_cone(cone: geometry.Cone, thickness, **kwargs):
    """
    Plot two cones separated by a distance `thickness`.

    Parameters: same as plot_cone, plus `thickness`.
    """
    cone_1, cone_2 = geometry.thick_cone_cones(cone, thickness)
    plot_cone(cone_1, **kwargs)
    kwargs.pop('color', None)
    fig = plot_cone(cone_2, color='blue', **kwargs)
    return fig


def plot_cone_fit(fit_result, **kwargs):
    """Plot the cone resulting from a cone fit to a point set."""
    cone = geometry.Cone(*fit_result['x'][:4], base_pos=geometry.Point(*fit_result['x'][4:]))
    fig = plot_cone(cone, **kwargs)
    return fig


def plot(data, *args, **kwargs):
    """
    Wraps plotting functions for use in Jupyter notebooks.

    Based on the type of `data`, this function will in turn call the following:

    - `pandas.DataFrame` or `vaex.DataFrame`: `ipv_plot_df`.
    - `geometry.Plane`: `plot_plane`.
    - `geometry.Cone`: `plot_cone`.

    See the documentation of those functions for how to call `plot`. All
    arguments and keyword arguments are passed on to the wrapped functions.
    """
    import pandas as pd
    import vaex

    if isinstance(data, (pd.DataFrame, vaex.dataframe.DataFrame)):
        ipv_plot_df(data, *args, **kwargs)
    elif isinstance(data, geometry.Plane):
        plot_plane(data, *args, **kwargs)
    elif isinstance(data, geometry.Cone):
        plot_cone(data, *args, **kwargs)
    else:
        raise TypeError("Type of `data` not supported.")


def clear():
    """Call ipyvolume.clear()."""
    ipv.clear()


def show():
    """Call `ipyvolume.show()`."""
    ipv.show()
