"""Offline YAM mesh calibration using the existing Viser alignment and ICP tools."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from lerobot_3d.icp import refine_icp_multiscale
from lerobot_3d.point_clouds.yam_robot_state import transform_points


def rigid_transform(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4x4 transform")
    rotation = matrix[:3, :3]
    if not (
        np.allclose(matrix[3], [0, 0, 0, 1])
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        and np.isclose(np.linalg.det(rotation), 1)
    ):
        raise ValueError("Expected a proper rigid transform")
    return matrix


def point_cloud(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.isfinite(points).all():
        raise ValueError("Expected at least 100 finite masked XYZ points in metres")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return cloud


def fit_mesh(camera_points, robot_points, initial):
    """Local refinement only; the caller supplies a manually aligned initial pose."""
    source, target = point_cloud(camera_points), point_cloud(robot_points)
    target.estimate_normals()
    result = refine_icp_multiscale(source, target, rigid_transform(initial))
    if result.fitness < 0.25:
        raise ValueError("Insufficient ICP overlap; inspect mask and manual alignment")
    return {
        "X_WC": rigid_transform(result.transformation).tolist(),
        "fitness": float(result.fitness),
        "inlier_rmse_m": float(result.inlier_rmse),
    }


def interactive_fit(viewer, camera_points, robot_points, initial):
    """Show the existing manual/ICP/confirm workflow; output is always a candidate."""
    point_cloud(camera_points)
    point_cloud(robot_points)
    viewer.show("/target", robot_points, None)
    manual = viewer.align(camera_points, rigid_transform(initial), title="Align masked camera cloud to YAM")
    fit = fit_mesh(camera_points, robot_points, manual)
    chosen = viewer.align(
        camera_points,
        np.asarray(fit["X_WC"]),
        T_fallback=manual,
        title="Review ICP; Abort keeps manual alignment",
    )
    return {
        "status": "candidate",
        "X_WC": rigid_transform(chosen).tolist(),
        "icp_proposal": fit,
        "note": "ICP metrics describe the proposal, not later manual adjustments. Validate on separate poses.",
    }


def camera_pose(candidate, states, observations):
    """Move a wrist calibration with measured FK; overhead stays in world."""
    if candidate["camera"] == "overhead":
        return rigid_transform(candidate["X_WC"])
    side = candidate["camera"]
    return states[side].get_link_transform(observations[side], side + "_gripper") @ rigid_transform(
        candidate["T_gripper_camera"]
    )


def mesh_residual(camera_points, robot_points, world_camera):
    """Held-out nearest-surface diagnostic without updating the fitted transform."""
    moved = point_cloud(transform_points(rigid_transform(world_camera), np.asarray(camera_points)))
    distances = np.asarray(moved.compute_point_cloud_distance(point_cloud(robot_points)))
    return {
        "count": len(distances),
        "median_distance_m": float(np.median(distances)),
        "p95_distance_m": float(np.quantile(distances, 0.95)),
        "status": "diagnostic_only",
        "note": "Partial overlap and nearby wrong links can give misleading residuals; inspect Viser.",
    }
