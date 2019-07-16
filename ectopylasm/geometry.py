"""Calculations on shapes and vectors."""

import numpy as np
import sympy as sy
import tqdm

import logging

logger = logging.getLogger('ectopylasm.geometry')
logger.setLevel(logging.INFO)


def normalize_vector(n):
    """Input vector `n` divided by its absolute size yields a vector of size 1."""
    n_size = n[0] * n[0] + n[1] * n[1] + n[2] * n[2]
    n = (n[0] / n_size, n[1] / n_size, n[2] / n_size)
    return n


def angle_between_two_vectors(a, b):
    """Calculate the angle in radians between two vectors `a` and `b`."""
    return np.arccos(np.sum(a * b) / np.sqrt(np.sum(a**2)) / np.sqrt(np.sum(b**2)))


def plane_surface(p, n, x_lim, z_lim, d=None):
    """
    Get plane surface coordinates.

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
    Convert plane point to two thick plane points.

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
    Generate a point in a plane.

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
    """
    Select the points that are within the thick plane.

    points_xyz: a vector of shape (3, N) representing N points in 3D space
    plane_point: a point in the plane
    plane_normal: the normal vector to the plane (x, y, z; any iterable)
    plane_thickness: the thickness of the plane (the distance between the two
                     composing planes)
    d [optional]: the constant in the plane equation ax + by + cz + d = 0; if
                  specified, `plane_point` will be ignored
    """
    if d is not None:
        plane_point = plane_point_from_d(plane_normal, d)
    point1, point2 = thick_plane_points(plane_point, plane_normal, plane_thickness)
    plane1 = sy.geometry.Plane(sy.geometry.Point3D(point1), normal_vector=plane_normal)
    plane2 = sy.geometry.Plane(sy.geometry.Point3D(point2), normal_vector=plane_normal)

    p_filtered = []
    for p_i in tqdm.tqdm(points_xyz.T):
        sy_point_i = sy.geometry.Point3D(tuple(p_i))
        if plane1.distance(sy_point_i) <= plane_thickness and plane2.distance(sy_point_i) <= plane_thickness:
            p_filtered.append(p_i)
    return p_filtered


def cone_surface(height, radius, rot_x=2 * np.pi, rot_y=2 * np.pi,
                 base_pos=(0, 0, 0), n_steps=20):
    """
    Calculate coordinates of the surface of a cone.

    height: height along the cone's central axis
    radius: radius of the circle
    rot_x: rotation angle about the x axis (radians)
    rot_y: rotation angle about the y axis (radians)
    base_pos: translation of base of cone to this position, iterable of three numbers
    n_steps: number of steps in the parametric range used for drawing (more gives a
             smoother surface, but may render more slowly)
    """
    h, r, u, theta = sy.symbols('h, r, u, theta')
    x_eqn = (h - u) / h * r * sy.cos(theta)
    y_eqn = (h - u) / h * r * sy.sin(theta)
    z_eqn = u

    x_rot_x = x_eqn
    y_rot_x = y_eqn * sy.cos(rot_x) - z_eqn * sy.sin(rot_x)
    z_rot_x = y_eqn * sy.sin(rot_x) + z_eqn * sy.cos(rot_x)

    x_rot_y = x_rot_x * sy.cos(rot_y) - z_rot_x * sy.sin(rot_y) + base_pos[0]
    y_rot_y = y_rot_x + base_pos[1]
    z_rot_y = x_rot_x * sy.sin(rot_y) + z_rot_x * sy.cos(rot_y) + base_pos[2]

    # get box limits in two dimensions
    u_steps = np.linspace(0, height, n_steps)
    theta_steps = np.linspace(0, 2 * np.pi, n_steps)
    u_array, theta_array = np.meshgrid(u_steps, theta_steps)

    # find corresponding y coordinates
    x, y, z = [], [], []
    for ui, thetai in zip(u_array.flatten(), theta_array.flatten()):
        x.append(float(x_rot_y.evalf(subs={h: height, r: radius, u: ui, theta: thetai})))
        y.append(float(y_rot_y.evalf(subs={h: height, r: radius, u: ui, theta: thetai})))
        z.append(float(z_rot_y.evalf(subs={h: height, r: radius, u: ui, theta: thetai})))

    return (np.array(x).reshape(u_array.shape),
            np.array(y).reshape(u_array.shape),
            np.array(z).reshape(u_array.shape))


def cone_axis_from_rotation(rot_x, rot_y):
    """Get the cone's axis unit vector from its rotation angles (radians)."""
    # z-unit vector (0, 0, 1) rotated twice
    cone_axis = (0, -np.sin(rot_x), np.cos(rot_x))  # rotation around x-axis
    cone_axis = np.array((-np.sin(rot_y) * cone_axis[2],
                          cone_axis[1],
                          np.cos(rot_y) * cone_axis[2]))  # around y
    return cone_axis


def thick_cone_base_positions(height, radius, thickness, rot_x, rot_y, base_pos):
    """
    Convert cone base position to two thick cone base positions.

    Given the cone parameters, return two base positions along the cone axis
    that are a certain distance apart, such that the distance between the
    cone surfaces (the directrices) is `thickness` apart.

    height: height along the cone's central axis
    radius: radius of the circle
    thickness: distance between the two cone surfaces (i.e. their directrices)
    rot_x: rotation angle about the x axis (radians)
    rot_y: rotation angle about the y axis (radians)
    base_pos: translation of base of cone to this position, iterable of three numbers
    """
    thickness = abs(thickness)
    base_distance = thickness / radius * height * np.sqrt(1 + radius**2 / height**2)  # trigonometry

    cone_axis = cone_axis_from_rotation(rot_x, rot_y)

    base_pos_1 = np.array(base_pos) - cone_axis * 0.5 * base_distance
    base_pos_2 = np.array(base_pos) + cone_axis * 0.5 * base_distance

    return base_pos_1, base_pos_2


def cone_apex_position(height, rot_x=2 * np.pi, rot_y=2 * np.pi, base_pos=(0, 0, 0)):
    """
    Get cone apex position from cone parameters.

    height: height along the cone's central axis
    rot_x: rotation angle about the x axis (radians)
    rot_y: rotation angle about the y axis (radians)
    base_pos: translation of base of cone to this position, iterable of three numbers
    """
    cone_axis = cone_axis_from_rotation(rot_x, rot_y)
    return np.array(base_pos) + cone_axis * height


def cone_opening_angle(height, radius):
    """Twice the opening angle is the maximum angle between directrices."""
    return np.arctan(radius / height)


def point_distance_to_cone(point, height, radius,
                           rot_x=2 * np.pi, rot_y=2 * np.pi, base_pos=(0, 0, 0),
                           return_extra=False):
    """
    Get distance of point to cone.

    Check whether for a point `point`, the shortest path to the cone is
    perpendicular to the cone surface (and if so, return it). If
    not, it is either "above" the apex and the shortest path is simply
    the line straight to the apex, or it is "below" the base, and the
    shortest path is the shortest path to the directrix (the base
    circle).

    This function returns a second value depending on which of the
    three above cases is true for point `point`. If we're using the
    perpendicular, it is True, if we're above the apex it is False and
    if it is below the base, it is None.

    Extra values can be returned to be reused outside the function by
    setting return_extra to True.
    """
    cone_axis = cone_axis_from_rotation(rot_x, rot_y)
    apex_pos = cone_apex_position(height, rot_x=rot_x, rot_y=rot_y, base_pos=base_pos)
    point_apex_vec = np.array(point) - apex_pos
    point_apex_angle = np.pi - angle_between_two_vectors(cone_axis, point_apex_vec)
    opening_angle = cone_opening_angle(height, radius)

    # for the second conditional, we need the length of the component of the
    # difference vector between P and apex along the closest generatrix
    point_apex_generatrix_angle = point_apex_angle - opening_angle
    point_apex_distance = np.sqrt(np.sum(point_apex_vec**2))
    point_apex_generatrix_component = point_apex_distance * np.cos(point_apex_generatrix_angle)
    generatrix_length = np.sqrt(radius**2 + height**2)

    returnees = {}
    if return_extra:
        returnees['opening_angle'] = opening_angle
        returnees['point_apex_angle'] = point_apex_angle

    if point_apex_angle > opening_angle + np.pi / 2:
        # "above" the apex
        return point_apex_distance, False, returnees
    elif point_apex_generatrix_component > generatrix_length:
        # "below" the directrix
        # use cosine rule to find length of third side
        return np.sqrt(point_apex_distance**2 + generatrix_length**2
                       - 2 * point_apex_distance * generatrix_length
                       * np.cos(point_apex_generatrix_angle)), None, returnees
    else:
        # "perpendicular" to a generatrix
        return point_apex_distance * np.sin(point_apex_generatrix_angle), True, returnees


def filter_points_cone(points_xyz, height, radius, thickness,
                       rot_x=2 * np.pi, rot_y=2 * np.pi, base_pos=(0, 0, 0)):
    """
    Select the points that are within the thick cone.

    points_xyz: a vector of shape (3, N) representing N points in 3D space
    height: height along the cone's central axis
    radius: radius of the circle
    thickness: distance between the two cone surfaces (i.e. their directrices)
    rot_x: rotation angle about the x axis (radians)
    rot_y: rotation angle about the y axis (radians)
    base_pos: translation of base of cone to this position, iterable of three numbers
    """
    base_pos_1, base_pos_2 = thick_cone_base_positions(height, radius, thickness, rot_x, rot_y, base_pos)

    p_filtered = []
    for p_i in tqdm.tqdm(points_xyz.T):
        d_cone1, flag_cone1, vals1 = point_distance_to_cone(p_i, height, radius,
                                                            rot_x=rot_x, rot_y=rot_y,
                                                            base_pos=base_pos_1, return_extra=True)
        d_cone2, flag_cone2, _ = point_distance_to_cone(p_i, height, radius,
                                                        rot_x=rot_x, rot_y=rot_y,
                                                        base_pos=base_pos_2, return_extra=True)
        if flag_cone2 is False or flag_cone1 is None:
            # it is definitely outside of the cones' range
            logger.debug(f"case 1: {p_i} was ignored")
            pass
        elif flag_cone1 is False:
            # the first condition is logically enclosed in the second, but the
            # first is faster and already covers a large part of the cases/volume:
            if abs(d_cone1) <= thickness or \
               abs(d_cone1) <= thickness / np.cos(vals1['point_apex_angle'] - vals1['opening_angle'] - np.pi / 2):
                p_filtered.append(p_i)
                logger.debug(f"case 2: {p_i} was added")
            else:
                logger.debug(f"case 3: {p_i} was ignored")
                pass
        elif abs(d_cone1) <= thickness and abs(d_cone2) <= thickness:
            p_filtered.append(p_i)
            logger.debug(f"case 4: {p_i} was added")
        else:
            logger.debug(f"case 5: {p_i} was ignored")
    return p_filtered
