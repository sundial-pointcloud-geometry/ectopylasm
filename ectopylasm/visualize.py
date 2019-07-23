"""Visualization of point cloud data and geometrical shapes."""

import numpy as np
import logging

import ipyvolume as ipv
import pptk

from ectopylasm import geometry


logger = logging.getLogger('ectopylasm.visualize')
logger.setLevel(logging.INFO)


def random_sample(xyz, total, sample_frac):
    """
    Get a random sample from a point cloud.

    xyz: array of shape (3, N) representing N points in 3D space
    total: number of points in xyz
    sample_frac: fraction of the total set that you want to return
    """
    sample = np.random.choice(total, int(sample_frac * total), replace=False)
    logger.debug("sample size:", int(sample_frac * total), "out of total", total)
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
    pptk.viewer(np.array([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T, **kwargs)


def ipv_plot_df(df, sample_frac=1, marker='circle_2d', **kwargs):
    """Plot vertices in a dataframe using ipyvolume."""
    if sample_frac < 1:
        xyz = random_sample(df, len(df), sample_frac)
    else:
        xyz = dict(x=df['x'].values, y=df['y'].values, z=df['z'].values)
    fig = ipv.scatter(**xyz, marker=marker, **kwargs)
    return fig


def pptk_plot_df(df, **kwargs):
    """Plot vertices in a dataframe using pptk."""
    pptk.viewer(np.array([df['x'], df['y'], df['z']]).T, **kwargs)


def plot_plane(plane: geometry.Plane, x_lim=None, z_lim=None, **kwargs):
    """
    Draw a plane.

    The limited coordinates are called x and z, corresponding to the first and
    third components of `p` and `n`. The final y coordinate is calculated
    based on the equation for a plane.

    plane: a Plane object
    x_lim [optional]: iterable of the two extrema in the x direction
    z_lim [optional]: same as x, but for z
    """
    if x_lim is None:
        x_lim = ipv.pylab.gcf().xlim
    if z_lim is None:
        z_lim = ipv.pylab.gcf().zlim
    fig = ipv.plot_surface(*geometry.plane_surface(plane, x_lim, z_lim), **kwargs)
    return fig


def plot_thick_plane(plane: geometry.Plane, thickness=0, d=None, **kwargs):
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
        p1, p2 = geometry.thick_plane_planes(plane, thickness)

        plot_plane(p1, **kwargs)
        fig = plot_plane(p2, **kwargs)
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
