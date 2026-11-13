"""Conservative landmark surface interpolation shared by the Blender fitting job.

Coordinates are approximate landmark geometry, never calibrated camera depth.
All fitting references are taken from the hash-bound identity measurement report.
"""
from __future__ import annotations

import numpy as np


def anatomical_frame(points):
    points = np.asarray(points, dtype=float)
    origin = (points[234] + points[454]) / 2
    right = points[454] - points[234]
    right /= np.linalg.norm(right)
    up = points[10] - points[152]
    height = np.linalg.norm(up)
    up -= right * np.dot(up, right)
    up /= np.linalg.norm(up)
    front = np.cross(right, up)
    if np.dot(front, points[1] - origin) < 0:
        front *= -1
    axes = np.stack([right, front, up], axis=1)
    if not np.isfinite(axes).all() or height < 1e-6:
        raise ValueError('Degenerate landmark frame')
    return origin, axes, height


def measured_surface(report):
    if report.get('schema_version') != 'avatar-identity-measurements.v1' or report.get('persona_id') != 'jensen-huang':
        raise ValueError('Expected hash-bound Jensen identity measurements')
    families = {}
    for row in report['accepted']:
        points = np.asarray(row['pose_normalized_points'][:468], dtype=float)
        if points.shape != (468, 3) or not np.isfinite(points).all():
            raise ValueError('Expected 468 finite face geometry points')
        origin, axes, height = anatomical_frame(points)
        families.setdefault(row['event_family'], []).append((points - origin) @ axes / height)
    if len(families) < 2:
        raise ValueError('Dense fitting requires multiple source event families')
    per_family = np.stack([np.median(rows, axis=0) for rows in families.values()])
    median = np.median(per_family, axis=0)
    spread = np.median(np.linalg.norm(per_family - median, axis=2), axis=0)
    return median, spread


class SmoothSurfaceFit:
    """Gaussian interpolation with uncertainty regularization and bounded offsets."""

    def __init__(self, anchors, targets, *, sigma, maximum, uncertainty):
        self.anchors = np.asarray(anchors, dtype=float)
        self.sigma = float(sigma)
        self.maximum = float(maximum)
        differences = np.asarray(targets, dtype=float) - self.anchors
        magnitude = np.linalg.norm(differences, axis=1)
        differences *= np.minimum(1, self.maximum / np.maximum(magnitude, 1e-12))[:, None]
        distance = np.sum((self.anchors[:, None] - self.anchors[None, :]) ** 2, axis=2)
        kernel = np.exp(-distance / (2 * self.sigma ** 2))
        regularization = .025 + np.asarray(uncertainty) * 2
        self.coefficients = np.linalg.solve(kernel + np.diag(regularization), differences)
        if not np.isfinite(self.coefficients).all():
            raise ValueError('Surface interpolation is not finite')

    def displacement(self, points):
        points = np.asarray(points, dtype=float)
        result = np.empty_like(points)
        # Bounds working memory to a few MiB rather than all mesh × landmark pairs.
        for start in range(0, len(points), 512):
            batch = points[start:start + 512]
            distance = np.sum((batch[:, None] - self.anchors[None, :]) ** 2, axis=2)
            offset = np.exp(-distance / (2 * self.sigma ** 2)) @ self.coefficients
            length = np.linalg.norm(offset, axis=1)
            offset *= np.minimum(1, self.maximum / np.maximum(length, 1e-12))[:, None]
            result[start:start + len(batch)] = offset
        return result
