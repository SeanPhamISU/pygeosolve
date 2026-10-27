"""Geometry."""

import abc
from decimal import DivisionByZero
import numpy as np
from .util import map_angle_about_zero


class Primitive(metaclass=abc.ABCMeta):
    """A primitive shape.

    Parameters
    ----------
    name : :class:`str`
        The name.

    points : sequence
        The points that make up the primitive.
    """

    def __init__(self, name, points):
        self.name = name
        for i, point in enumerate(points):
            if not isinstance(point, Point3D):
                points[i] = Point3D(f"__{self.name}_p{i}__", *point)

        # Quick check that names are unique.
        assert len(set(p.name for p in points)) == len(points)

        self.points = points

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.__class__.__name__}@{hex(id(self))}>"

    def validate(self):
        return True


class Line3D(Primitive):
    """A line formed between two 2D points in Euclidean space.

    Parameters
    ----------
    start, end : :class:`tuple` containg two :class:`floats <float>` or
                 :class:`points <.Point>`
    """

    def __init__(self, name, start, end):
        super().__init__(name, [start, end])

    @property
    def start(self):
        return self.points[0]

    @property
    def end(self):
        return self.points[1]

    def dx(self):
        """The difference between the end and start x-coordinates.

        Returns
        -------
        :class:`float`
            The difference.
        """
        return self.end.x - self.start.x

    def dy(self):
        """The difference between the end and start y-coordinates.

        Returns
        -------
        :class:`float`
            The difference.
        """
        return self.end.y - self.start.y

    def dz(self):
        """The difference between the end and start z-coordinates.

        Returns
        -------
        :class:`float`
            The difference.
        """
        return self.end.z - self.start.z

    def length(self):
        """The line length.

        Returns
        -------
        :class:`float`
            The length.
        """
        return (self.end - self.start).norm()


    def _direction_vector(self):
        direction = np.array([self.dx(), self.dy(), self.dz()], dtype=float)
        if np.allclose(direction, 0):
            raise ValueError("angle is undefined for a zero-length line")
        return direction

    def angle(self, normal=None):
        """The angle of the line's direction in a right-handed coordinate system.

        By default the signed angle is measured in the XY plane about the +Z axis,
        which preserves the project's clockwise convention for 2D work while
        supporting 3D direction vectors.

        Parameters
        ----------
        normal : array-like, optional
            The normal vector of the plane in which the signed angle should be
            measured. If omitted, the +Z axis is used.
        """
        direction = self._direction_vector()
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0])
        normal = np.asarray(normal, dtype=float)
        if np.allclose(normal, 0):
            raise ValueError("normal must be non-zero")

        normal /= np.linalg.norm(normal)
        reference = np.array([1.0, 0.0, 0.0])
        if np.isclose(np.linalg.norm(np.cross(reference, normal)), 0):
            reference = np.array([0.0, 1.0, 0.0])
        normal_component = np.cross(direction, reference)
        angle = np.degrees(np.arctan2(np.dot(normal_component, normal), np.dot(direction, reference)))
        return map_angle_about_zero(angle)

    def angle_to(self, other, normal=None):
        """The signed clockwise angle from `self` to `other`.

        The angle is defined as the clockwise rotation from the direction of `self` to
        get to the direction of `other`, measured in the plane whose normal is
        `normal`. Directions are projected onto this plane before measuring the angle.
        When no normal is supplied, the +Z axis is used, preserving the library's
        existing XY-plane convention.

        Parameters
        ----------
        other : :class:`.Line`
            The other line.
        normal : array-like, optional
            The normal of the plane in which the angle is measured.
        """
        if not isinstance(other, Line3D):
            raise TypeError("other must be a Line3D instance")
        line1_len = self.length()
        line2_len = other.length()
        dot_prod = self.dx() * other.dx() + self.dy() * other.dy() + self.dz() * other.dz()
        angle = np.degrees( np.arccos( dot_prod / (line1_len * line2_len) ) )

        return map_angle_about_zero(angle)

    def validate(self):
        zerolength = np.isclose(self.length(), 0)

        if zerolength:
            return Invalid(self, "zero length")

        return True

    def __str__(self):
        points = ", ".join(str(point) for point in self.points)
        return (
            f"{self.__class__.__name__}({self.name}, [{points}], "
            f"length={self.length()}, angle={self.angle()})"
        )


class Plane3D(Primitive):
    """A plane defined by three non-collinear 3D points."""

    def __init__(self, name, point_a, point_b, point_c):
        super().__init__(name, [point_a, point_b, point_c])

    def normal(self):
        """Return the plane's unnormalized normal vector."""
        point_a, point_b, point_c = self.points
        vector_a = np.array(
            [point_b.x - point_a.x, point_b.y - point_a.y, point_b.z - point_a.z]
        )
        vector_b = np.array(
            [point_c.x - point_a.x, point_c.y - point_a.y, point_c.z - point_a.z]
        )
        return np.cross(vector_a, vector_b)

    def validate(self):
        point_a, point_b, point_c = self.points
        vector_a = np.array(
            [point_b.x - point_a.x, point_b.y - point_a.y, point_b.z - point_a.z]
        )
        vector_b = np.array(
            [point_c.x - point_a.x, point_c.y - point_a.y, point_c.z - point_a.z]
        )
        scale = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if np.linalg.norm(self.normal()) <= np.finfo(float).eps * scale:
            return Invalid(self, "collinear points do not define a plane")
        return True

    def __str__(self):
        points = ", ".join(str(point) for point in self.points)
        return f"{self.__class__.__name__}({self.name}, [{points}])"


class Point3D(Primitive):
    """A 3D point in Euclidean space.

    Normally points should not be instantiated directly, but via :class:`primitives
    <.Primitive>`.

    Parameters
    ----------
    name : :class:`str`
        The name of this point.

    x, y, z : :class:`float`
        The x, y, and z coordinates.
    """

    def __init__(self, name, x, y, z):
        super().__init__(name, [self])
        self.params = [x, y, z]

    @property
    def x(self):
        return self.params[0]

    @property
    def y(self):
        return self.params[1]

    @property
    def z(self):
        return self.params[2]

    def norm(self):
        return np.sqrt(np.power(self.x, 2) + np.power(self.y, 2) + np.power(self.z, 2))

    def __add__(self, other):
        return self.__class__(
            self._op_name("+", other), self.x + other.x, self.y + other.y, self.z + other.z
        )

    def __sub__(self, other):
        return self.__class__(
            self._op_name("-", other), self.x - other.x, self.y - other.y, self.z - other.z
        )

    def __str__(self):
        return f"{self.__class__.__name__}({self.name}, ({self.x}, {self.y}, {self.z}))"

    def _op_name(self, op, other):
        return f"{self}{op}{other}"

    def position(self, P):
        self.params[0] = P[0]
        self.params[1] = P[1]
        self.params[2] = P[2]

    def get_tuple(self):
        return tuple(self.params)

class Invalid:
    def __init__(self, primitive, reason):
        self.primitive = primitive
        self.reason = reason

    def __str__(self):
        return f"{self.primitive} ({self.reason})"
