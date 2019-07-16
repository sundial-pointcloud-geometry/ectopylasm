import numpy as np
import logging

import ipyvolume as ipv
import pptk

from ectopylasm import geometry


logger = logging.getLogger('ectopylasm.visualize')
logger.setLevel(logging.INFO)


def random_sample(xyz, total, sample_frac):
    sample = np.random.choice(total, int(sample_frac * total), replace=False)
    logger.debug("sample size:", int(sample_frac * total), "out of total", total)
    return dict(x=xyz['x'][sample], y=xyz['y'][sample], z=xyz['z'][sample])


def ipv_plot_plydata(plydata, sample_frac=1, shape='circle2d', **kwargs):
    if sample_frac < 1:
        xyz = random_sample(plydata['vertex'], plydata['vertex'].count, sample_frac)
    else:
        xyz = dict(x=plydata['vertex']['x'], y=plydata['vertex']['y'], z=plydata['vertex']['z'])
    fig = ipv.scatter(**xyz, shape=shape, **kwargs)
    return fig


def pptk_plot_plydata(plydata, **kwargs):
    pptk.viewer(np.array([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T, **kwargs)


def ipv_plot_df(df, sample_frac=1, marker='circle_2d', **kwargs):
    if sample_frac < 1:
        xyz = random_sample(df, len(df), sample_frac)
    else:
        xyz = dict(x=df['x'].values, y=df['y'].values, z=df['z'].values)
    fig = ipv.scatter(**xyz, marker=marker, **kwargs)
    return fig


def pptk_plot_df(df, **kwargs):
    pptk.viewer(np.array([df['x'], df['y'], df['z']]).T, **kwargs)


def plot_plane(p, n, x_lim=None, z_lim=None, d=None):
    """
    Draw a plane.

    The limited coordinates are called x and z, corresponding to the first and
    third components of `p` and `n`. The final y coordinate is calculated
    based on the equation for a plane.

    p: a point in the plane (x, y, z; any iterable)
    n: the normal vector to the plane (x, y, z; any iterable)
    x_lim [optional]: iterable of the two extrema in the x direction
    z_lim [optional]: same as x, but for z
    d [optional]: if d is known (in-product of p and n), then this can be
                  supplied directly; p is disregarded in this case.
    """
    if x_lim is None:
        x_lim = ipv.pylab.gcf().xlim
    if z_lim is None:
        z_lim = ipv.pylab.gcf().zlim
    fig = ipv.plot_surface(*geometry.plane_surface(p, n, x_lim, z_lim, d=d))
    return fig


def plot_thick_plane(p, n, thickness=0, d=None, **kwargs):
    """
    Draw two co-planar planes, separated by a distance `thickness`.

    p: a point in the plane (x, y, z; any iterable)
    n: the normal vector to the plane (x, y, z; any iterable)
    thickness: the distance between the two co-planar planes
    x_lim [optional]: iterable of the two extrema in the x direction
    z_lim [optional]: same as x, but for z
    d [optional]: if d is known (in-product of p and n), then this can be
                  supplied directly; p is disregarded in this case.
    """
    if thickness <= 0:
        fig = plot_plane(p, n, d=d, **kwargs)
    else:
        if d is not None:
            p = geometry.plane_point_from_d(n, d)
        # find points in the two planes and plot them
        p1, p2 = geometry.thick_plane_points(p, n, thickness)

        plot_plane(p1, n, d=d, **kwargs)
        fig = plot_plane(p2, n, d=d, **kwargs)
    return fig


def plot_plane_fit(fit_result, **kwargs):
    p_fit = fit_result.params
    fig = plot_plane(None, (p_fit['a'], p_fit['b'], p_fit['c']), d=p_fit['d'], **kwargs)
    return fig


def plot_cone(height, radius, rot_x=2*np.pi, rot_y=2*np.pi, base_pos=(0, 0, 0), n_steps=20, **kwargs):
    """
    Draw a cone surface.

    height: height along the z-axis
    radius: radius of the circle
    rot_x: rotation angle about the x axis (radians)
    rot_y: rotation angle about the y axis (radians)
    base_pos: translation of base of cone to this position, iterable of three numbers
    n_steps: number of steps in the parametric range used for drawing (more gives a
             smoother surface, but may render more slowly)
    """
    fig = ipv.plot_surface(*geometry.cone_surface(height, radius, rot_x=rot_x, rot_y=rot_y,
                                                  base_pos=base_pos, n_steps=n_steps),
                           **kwargs)
    return fig


def plot_thick_cone(height, radius, thickness,
                    rot_x=2*np.pi, rot_y=2*np.pi, base_pos=(0, 0, 0),
                    **kwargs):
    """
    Plot two cones separated by a distance `thickness`.

    Parameters: same as plot_cone, plus `thickness`.
    """
    base_pos_1, base_pos_2 = geometry.thick_cone_base_positions(height, radius, thickness, rot_x, rot_y, base_pos)
    plot_cone(height, radius, rot_x=rot_x, rot_y=rot_y, base_pos=base_pos_1, **kwargs)
    kwargs.pop('color', None)
    fig = plot_cone(height, radius, rot_x=rot_x, rot_y=rot_y, base_pos=base_pos_2, color='blue', **kwargs)
    return fig
