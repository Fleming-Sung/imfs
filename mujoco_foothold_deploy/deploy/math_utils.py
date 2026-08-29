"""Small scalar-first quaternion helpers used by the deployment loop."""

import numpy as np


def quat_conjugate(q):
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply(a, b):
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def rotate(q, vector):
    pure = np.r_[0.0, np.asarray(vector, dtype=np.float64)]
    return quat_multiply(quat_multiply(q, pure), quat_conjugate(q))[1:]


def rotate_inverse(q, vector):
    return rotate(quat_conjugate(q), vector)


def canonicalize(q):
    q = np.asarray(q, dtype=np.float64)
    return -q if q[0] < 0.0 else q


def yaw_quaternion(yaw):
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)
