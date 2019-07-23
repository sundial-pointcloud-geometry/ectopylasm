"""Calculations on shapes and vectors."""

import numpy as np
import sympy as sy
import tqdm

import logging
import dataclasses
import typing
import enum

logger = logging.getLogger('ectopylasm.geometry')
logger.setLevel(logging.INFO)


def normalize_vector(vector):
    """Input `vector` divided by its absolute size yields a vector of size 1."""
    return vector / np.linalg.norm(vector)


def angle_between_two_vectors(a, b):
    """
    Calculate the angle in radians between two vectors `a` and `b`.

    Implementation credits to https://stackoverflow.com/a/13849249/1199693.
    """
    a_n = normalize_vector(a)
    b_n = normalize_vector(b)
    return np.arccos(np.clip(np.dot(a_n, b_n), -1.0, 1.0))


@dataclasses.dataclass(frozen=True)
class Point(object):
    """A three dimensional point with x, y and z components."""

    x: float
    y: float
    z: float

    def to_array(self):
        """Convert to a NumPy array `np.array((x, y, z))`."""
        return np.array((self.x, self.y, self.z))


@dataclasses.dataclass(frozen=True)
class Plane(object):
    """
    A plane.

    The plane is defined by four parameters, a, b, c and d, which form the
    plane equation a*x + b*y + c*z + d = 0. Points (x, y, z) for which this
    equation applies are points on the plane.

    On creation, the input parameters a, b and c are normalized. When seen
    as a vector n = (a, b, c), n is the normal vector to the plane,
    indicating its direction. This vector is normalized to have length one.

    The fourth parameter d relates to the position of the plane in space. It
    can be calculated from a known point in the plane (X, Y, Z) as
    d = -a*X - b*Y - c*Z, but can also be given directly.
    """

    a: float
    b: float
    c: float
    d: float

    def __init__(self, a, b, c, d):
        """
        Construct the plane, taking the four plane parameters as input.

        Normalizes a, b and c so that the vector n = (a, b, c) has length 1.
        """
        a, b, c = normalize_vector((a, b, c))
        object.__setattr__(self, 'a', a)
        object.__setattr__(self, 'b', b)
        object.__setattr__(self, 'c', c)
        object.__setattr__(self, 'd', d)

    @staticmethod
    def d(point, normal):
        """Calculate d factor in plane equation ax + by + cz + d = 0."""
        return -(point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2])

    @classmethod
    def from_point(cls, a, b, c, point):
        """Plane constructor that uses a point on the plane as input instead of d."""
        a, b, c = normalize_vector((a, b, c))
        return cls(a, b, c, cls.d(point, (a, b, c)))

    def generate_point(self):
        """
        Generate a point in the plane.

        Calculate a point in the plane based on d at x,y=0,0 (could be
        anywhere); -cz = ax + by + d. If c happens to be zero, try x,z=0,0, and
        if b is zero as well, do y,z=0,0.
        """
        if self.c != 0:
            return (0, 0, -self.d / self.c)
        elif self.b != 0:
            return (0, -self.d / self.b, 0)
        else:
            return (-self.d / self.a, 0, 0)


def plane_surface(plane: Plane, x_lim, z_lim):
    """
    Get plane surface coordinates.

    Calculate coordinates of the part of a plane inside a cubical box. The
    limited coordinates are called x and z, corresponding to the first and
    third components of `p` and `n`. The final y coordinate is calculated
    based on the equation for a plane.

    plane: a Plane object
    x_lim: iterable of the two extrema in the x direction
    z_lim: same as x, but for z
    """
    # get box limits in two dimensions
    x, z = np.meshgrid(x_lim, z_lim)

    # find corresponding y coordinates
    y = -(plane.a * x + plane.c * z + plane.d) / plane.b
    return x, y, z


def thick_plane_points(plane: Plane, thickness, plane_point=None):
    """
    Convert plane point to two thick plane points.

    Given a Plane and a thickness, return two points along the normal that
    are `thickness` apart. Optionally specify a specific point in the plane.
    """
    if plane_point is None:
        plane_point = plane.generate_point()

    p1 = (plane_point[0] + 0.5 * thickness * plane.a,
          plane_point[1] + 0.5 * thickness * plane.b,
          plane_point[2] + 0.5 * thickness * plane.c)
    p2 = (plane_point[0] - 0.5 * thickness * plane.a,
          plane_point[1] - 0.5 * thickness * plane.b,
          plane_point[2] - 0.5 * thickness * plane.c)
    return p1, p2


def thick_plane_planes(plane: Plane, thickness):
    """Convert plane to two planes separated by thickness."""
    plane_point_1, plane_point_2 = thick_plane_points(plane, thickness)
    plane_1 = Plane.from_point(plane.a, plane.b, plane.c, plane_point_1)
    plane_2 = Plane.from_point(plane.a, plane.b, plane.c, plane_point_2)
    return plane_1, plane_2


def point_distance_to_plane(point, plane: Plane):
    """
    Get signed distance of point to plane.

    The sign of the resulting distance tells you whether the point is in
    the same or the opposite direction of the plane normal vector.

    point: an iterable of length 3 representing a point in 3D space
    plane: a Plane object
    """
    # from http://mathworld.wolfram.com/Point-PlaneDistance.html
    # N.B.: no need to divide by ||(a,b,c)||, since that is always 1
    return (plane.a * point[0] + plane.b * point[1] + plane.c * point[2] + plane.d)


def filter_points_plane(points_xyz, plane: Plane, plane_thickness):
    """
    Select the points that are within the thick plane.

    points_xyz: a vector of shape (3, N) representing N points in 3D space
    plane: a Plane object
    plane_thickness: the thickness of the plane (the distance between the two
                     composing planes)
    """
    plane_1, plane_2 = thick_plane_planes(plane, plane_thickness)

    p_filtered = []
    for p_i in points_xyz.T:
        distance_1 = point_distance_to_plane(p_i, plane_1)
        distance_2 = point_distance_to_plane(p_i, plane_2)
        if abs(distance_1) <= plane_thickness and abs(distance_2) <= plane_thickness:
            p_filtered.append(p_i)
    return p_filtered


@dataclasses.dataclass(frozen=True)
class Cone(object):
    """
    A cone.

    The cone is defined mainly by its `height` and `radius`. When the other
    parameters are left at their default values, this will produce a cone with
    its axis along the z-axis and the center of its circular base at position
    (x, y, z) = (0, 0, 0).

    Three optional parameters define its location and orientation: two rotation
    parameters `rot_x` and `rot_y`, giving respectively rotations around the x
    and y axes (the x rotation is performed first, then the y rotation) and one
    translation parameter called `base_pos`, which itself is a `Point` object,
    and which moves the position of the circular base of the cone.
    """

    height: float
    radius: float
    rot_x: float = dataclasses.field(default=2 * np.pi, metadata={'unit': 'radians'})
    rot_y: float = dataclasses.field(default=2 * np.pi, metadata={'unit': 'radians'})
    base_pos: Point = Point(0, 0, 0)

    def axis(self):
        """Get the cone's axis unit vector from its rotation angles (radians)."""
        # z-unit vector (0, 0, 1) rotated twice
        cone_axis = (0, -np.sin(self.rot_x), np.cos(self.rot_x))  # rotation around x-axis
        cone_axis = np.array((-np.sin(self.rot_y) * cone_axis[2],
                              cone_axis[1],
                              np.cos(self.rot_y) * cone_axis[2]))  # around y
        return cone_axis

    def apex_position(self):
        """Get cone apex position from cone parameters."""
        return self.base_pos.to_array() + self.axis() * self.height

    def opening_angle(self):
        """Twice the opening angle is the maximum angle between directrices."""
        return np.arctan(self.radius / self.height)


def cone_surface(cone: Cone, n_steps=20):
    """
    Calculate coordinates of the surface of a cone.

    cone: a Cone object
    n_steps: number of steps in the parametric range used for drawing (more gives a
             smoother surface, but may render more slowly)
    """
    h, r, u, theta = sy.symbols('h, r, u, theta')
    x_eqn = (h - u) / h * r * sy.cos(theta)
    y_eqn = (h - u) / h * r * sy.sin(theta)
    z_eqn = u

    x_rot_x = x_eqn
    y_rot_x = y_eqn * sy.cos(cone.rot_x) - z_eqn * sy.sin(cone.rot_x)
    z_rot_x = y_eqn * sy.sin(cone.rot_x) + z_eqn * sy.cos(cone.rot_x)

    x_rot_y = x_rot_x * sy.cos(cone.rot_y) - z_rot_x * sy.sin(cone.rot_y) + cone.base_pos.x
    y_rot_y = y_rot_x + cone.base_pos.x
    z_rot_y = x_rot_x * sy.sin(cone.rot_y) + z_rot_x * sy.cos(cone.rot_y) + cone.base_pos.z

    # get box limits in two dimensions
    u_steps = np.linspace(0, cone.height, n_steps)
    theta_steps = np.linspace(0, 2 * np.pi, n_steps)
    u_array, theta_array = np.meshgrid(u_steps, theta_steps)

    # find corresponding y coordinates
    x, y, z = [], [], []
    for ui, thetai in zip(u_array.flatten(), theta_array.flatten()):
        x.append(float(x_rot_y.evalf(subs={h: cone.height, r: cone.radius, u: ui, theta: thetai})))
        y.append(float(y_rot_y.evalf(subs={h: cone.height, r: cone.radius, u: ui, theta: thetai})))
        z.append(float(z_rot_y.evalf(subs={h: cone.height, r: cone.radius, u: ui, theta: thetai})))

    return (np.array(x).reshape(u_array.shape),
            np.array(y).reshape(u_array.shape),
            np.array(z).reshape(u_array.shape))


def thick_cone_base_positions(cone: Cone, thickness):
    """
    Convert cone base position to two thick cone base positions.

    Given the cone parameters, return two base positions along the cone axis
    that are a certain distance apart, such that the distance between the
    cone surfaces (the directrices) is `thickness` apart.

    cone: a Cone object
    thickness: distance between the two cone surfaces (i.e. their directrices)
    """
    thickness = abs(thickness)
    # trigonometry:
    base_distance = thickness / cone.radius * cone.height * np.sqrt(1 + cone.radius**2 / cone.height**2)

    cone_axis = cone.axis()

    base_pos_bottom = cone.base_pos.to_array() - cone_axis * 0.5 * base_distance
    base_pos_top = cone.base_pos.to_array() + cone_axis * 0.5 * base_distance

    return base_pos_bottom, base_pos_top


def thick_cone_cones(cone: Cone, thickness) -> typing.Tuple[Cone, Cone]:
    """
    Convert one Cone to two cones separated by `thickness`.

    Given the cone parameters, return two cones, such that the distance between
    the cone surfaces (the directrices) is `thickness` apart.

    cone: a Cone object
    thickness: distance between the two cone surfaces (i.e. their directrices)
    """
    base_pos_bottom, base_pos_top = thick_cone_base_positions(cone, thickness)
    cone_bottom = Cone(cone.height, cone.radius, rot_x=cone.rot_x,
                       rot_y=cone.rot_y, base_pos=Point(*base_pos_bottom))
    cone_top = Cone(cone.height, cone.radius, rot_x=cone.rot_x,
                    rot_y=cone.rot_y, base_pos=Point(*base_pos_top))
    return cone_bottom, cone_top


class ConeRegion(enum.Enum):
    """
    Class defining three regions in and around cones.

    These regions are used in point_distance_to_cone to pass on information
    about which kind of region the point is in. This can be used in other
    functions (like filter_points_cone).

    The three options are:
    - perpendicular: the point is at a location where its shortest distance to
                     the cone surface is perpendicular to that surface
    - above_apex: the point is somewhere above the apex of the cone, but not
                  perpendicular to the surface
    - below_directrix: the point is not perpendicular to the surface and it is
                       below the directrix
    """

    perpendicular = enum. auto()
    above_apex = enum.auto()
    below_directrix = enum.auto()


def point_distance_to_cone(point, cone: Cone, return_extra=False):
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
    cone_axis = cone.axis()
    apex_pos = cone.apex_position()
    point_apex_vec = np.array(point) - apex_pos
    point_apex_angle = np.pi - angle_between_two_vectors(cone_axis, point_apex_vec)
    opening_angle = cone.opening_angle()

    # for the second conditional, we need the length of the component of the
    # difference vector between P and apex along the closest generatrix
    point_apex_generatrix_angle = point_apex_angle - opening_angle
    point_apex_distance = np.sqrt(np.sum(point_apex_vec**2))
    point_apex_generatrix_component = point_apex_distance * np.cos(point_apex_generatrix_angle)
    generatrix_length = np.sqrt(cone.radius**2 + cone.height**2)

    returnees = {}
    if return_extra:
        returnees['opening_angle'] = opening_angle
        returnees['point_apex_angle'] = point_apex_angle

    if point_apex_angle > opening_angle + np.pi / 2:
        # "above" the apex
        return point_apex_distance, ConeRegion.above_apex, returnees
    elif point_apex_generatrix_component > generatrix_length:
        # "below" the directrix
        # use cosine rule to find length of third side
        return np.sqrt(point_apex_distance**2 + generatrix_length**2
                       - 2 * point_apex_distance * generatrix_length
                       * np.cos(point_apex_generatrix_angle)), ConeRegion.below_directrix, returnees
    else:
        # "perpendicular" to a generatrix
        return point_apex_distance * np.sin(point_apex_generatrix_angle), ConeRegion.perpendicular, returnees


def filter_points_cone(points_xyz, cone: Cone, thickness):
    """
    Select the points that are within the thick cone.

    points_xyz: a vector of shape (3, N) representing N points in 3D space
    cone: a Cone object
    thickness: distance between the two cone surfaces (i.e. their directrices)
    """
    cone_bottom, cone_top = thick_cone_cones(cone, thickness)

    p_filtered = []
    for p_i in tqdm.tqdm(points_xyz.T):
        d_cone_bottom, flag_cone_bottom, vals_bottom = point_distance_to_cone(p_i, cone_bottom, return_extra=True)
        d_cone_top, flag_cone_top, _ = point_distance_to_cone(p_i, cone_top, return_extra=True)
        if flag_cone_top is ConeRegion.above_apex or flag_cone_bottom is ConeRegion.below_directrix:
            # it is definitely outside of the cones' range
            logger.debug(f"case 1: {p_i} was ignored")
            pass
        elif flag_cone_bottom is ConeRegion.above_apex:
            # the first condition is logically enclosed in the second, but the
            # first is faster and already covers a large part of the cases/volume:
            if abs(d_cone_bottom) <= thickness or \
               abs(d_cone_bottom) <= thickness / np.cos(vals_bottom['point_apex_angle'] - vals_bottom['opening_angle'] - np.pi / 2):
                p_filtered.append(p_i)
                logger.debug(f"case 2: {p_i} was added")
            else:
                logger.debug(f"case 3: {p_i} was ignored")
                pass
        elif abs(d_cone_bottom) <= thickness and abs(d_cone_top) <= thickness:
            p_filtered.append(p_i)
            logger.debug(f"case 4: {p_i} was added")
        else:
            logger.debug(f"case 5: {p_i} was ignored")
    return p_filtered
