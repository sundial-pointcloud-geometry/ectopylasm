import numpy as np
import logging

import ipyvolume as ipv
import pptk

import ectopylasm.geometry as geometry


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
    ipv.scatter(**xyz, shape=shape, **kwargs)
    ipv.show()


def pptk_plot_plydata(plydata, **kwargs):
    pptk.viewer(np.array([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T)


def ipv_plot_df(df, sample_frac=1, shape='circle2d', **kwargs):
    if sample_frac < 1:
        xyz = random_sample(df, len(df), sample_frac)
    else:
        xyz = dict(x=df['x'].values, y=df['y'].values, z=df['z'].values)
    ipv.scatter(**xyz, shape=shape, **kwargs)
    ipv.show()


def pptk_plot_df(df, **kwargs):
    pptk.viewer(np.array([df['x'], df['y'], df['z']]).T)


def plot_plane(p, n, x_lim, z_lim, d=None):
    """
    Draw a plane.
    The limited coordinates are called x and z, corresponding to the first and
    third components of `p` and `n`. The final y coordinate is calculated
    based on the equation for a plane.

    p: a point in the plane (x, y, z; any iterable)
    n: the normal vector to the plane (x, y, z; any iterable)
    x_lim: iterable of the two extrema in the x direction
    z_lim: same as x, but for z
    d [optional]: if d is known (in-product of p and n), then this can be
                  supplied directly; p is disregarded in this case.
    """
    ipv.plot_surface(*geometry.plane_surface(p, n, x_lim, z_lim, d=d))


def plot_thick_plane(p, n, x_lim, z_lim, thickness=0, d=None):
    """
    Draw two co-planar planes, separated by a distance `thickness`.

    p: a point in the plane (x, y, z; any iterable)
    n: the normal vector to the plane (x, y, z; any iterable)
    x_lim: iterable of the two extrema in the x direction
    z_lim: same as x, but for z
    thickness: the distance between the two co-planar planes
    d [optional]: if d is known (in-product of p and n), then this can be
                  supplied directly; p is disregarded in this case.
    """
    if thickness <= 0:
        plot_plane(p, n, x_lim, z_lim, d=d)
    else:
        if d is not None:
            p = geometry.plane_point_from_d(n, d)
        # find points in the two planes and plot them
        p1, p2 = geometry.thick_plane_points(p, n, thickness)

        plot_plane(p1, n, x_lim, z_lim, d=d)
        plot_plane(p2, n, x_lim, z_lim, d=d)
