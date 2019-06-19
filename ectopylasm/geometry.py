import numpy as np
from sympy.geometry import Plane as syPlane, Point3D as syPoint3D
import tqdm


def normalize_vector(n):
    n_size = n[0] * n[0] + n[1] * n[1] + n[2] * n[2]
    n = (n[0] / n_size, n[1] / n_size, n[2] / n_size)
    return n


def plane_surface(p, n, x_lim, z_lim, d=None):
    """
    Calculate coordinates of the part of a plane inside a cubical box. The
    limited coordinates are called x and z, corresponding to the first and
    third components of `p` and `n`. The final y coordinate is calculated
    based on the equation for a plane.

    p: a point in the plane (x, y, z; any iterable)
    n: the normal vector to the plane (x, y, z; any iterable)
    x_lim: iterable of the two extrema in the x direction
    z_lim: same as x, but for z
    d [optional]: if d is known (in-product of p and n), then this can be
                  supplied directly; p is disregarded in this case.
    """
    n = normalize_vector(n)

    if d is None:
        d = -(n[0] * p[0] + n[1] * p[1] + n[2] * p[2])

    # get box limits in two dimensions
    x, z = np.meshgrid(x_lim, z_lim)

    # find corresponding y coordinates
    y = -(n[0] * x + n[2] * z + d) / n[1]
    return x, y, z


def thick_plane_points(p, n, thickness):
    """
    Given a point, a normal vector and a thickness, return two points
    along the normal that are `thickness` apart.
    """
    n = normalize_vector(n)

    p1 = (p[0] + 0.5 * thickness * n[0],
          p[1] + 0.5 * thickness * n[1],
          p[2] + 0.5 * thickness * n[2])
    p2 = (p[0] - 0.5 * thickness * n[0],
          p[1] - 0.5 * thickness * n[1],
          p[2] - 0.5 * thickness * n[2])
    return p1, p2


def plane_point_from_d(n, d):
    """
    Calculate a point in the plane based on d at x,y=0,0 (could be
    anywhere); -cz = ax + by + d. If c happens to be zero, try x,z=0,0, and
    if b is zero as well, do y,z=0,0.

    n: the normal vector to the plane (x, y, z; any iterable)
    d: the constant in the plane equation ax + by + cz + d = 0
    """
    if n[2] != 0:
        return (0, 0, -d / n[2])
    elif n[1] != 0:
        return (0, -d / n[1], 0)
    else:
        return (-d / n[0], 0, 0)


def filter_points_plane(points_xyz, plane_point, plane_normal, plane_thickness, d=None):
    if d is not None:
        plane_point = plane_point_from_d(plane_normal, d)
    point1, point2 = thick_plane_points(plane_point, plane_normal, plane_thickness)
    plane1 = syPlane(syPoint3D(point1), normal_vector=plane_normal)
    plane2 = syPlane(syPoint3D(point2), normal_vector=plane_normal)

    p_filtered = []
    for p_i in tqdm.tqdm(points_xyz.T):
        sy_point_i = syPoint3D(tuple(p_i))
        if plane1.distance(sy_point_i) <= plane_thickness and plane2.distance(sy_point_i) <= plane_thickness:
            p_filtered.append(p_i)
    return p_filtered
