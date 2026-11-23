"""Patch-to-patch and point-to-patch visibility via ray casting with trimesh.

sparrowpy checks every patch pair against every patch as a possible
blocker, which scales with n_patches**3 and takes minutes for rooms with a
few thousand patches. Its check between a point (source or receiver) and
all patches scales with n_patches**2 per point. Casting one ray per pair
against a BVH of the room mesh (embree, if ``embreex`` is installed) gives
the same results in a fraction of a second. Without ``embreex`` (no wheels
for e.g. linux/arm64), trimesh falls back to its own ray casting with an
rtree, which gives the same results but is slower.

Two patches see each other if
1. they face each other: each one's center lies in front of the other's
   plane (this also excludes coplanar pairs), and
2. no other triangle intersects the segment between their centers.

A point sees a patch if it lies in front of the patch's plane and no
other triangle intersects the segment between the point and the patch's
center.
"""
import numpy as np
import trimesh

_CHUNK_SIZE = 1_000_000  # rays per batch, bounds the memory use
# same threshold as ``eta`` in sparrowpy's visibility check, for points
# lying in a patch's plane
_PLANE_TOLERANCE = 1e-6


def _room_mesh(patches_points: np.ndarray) -> trimesh.Trimesh:
    """Return a mesh with one face per patch, in the order of the patches."""
    n_patches = patches_points.shape[0]
    return trimesh.Trimesh(
        patches_points.reshape(-1, 3),
        np.arange(3 * n_patches).reshape(n_patches, 3),
        process=False)


def _blocked(
        mesh: trimesh.Trimesh, origins: np.ndarray, targets: np.ndarray,
        target_ids: np.ndarray, own_ids: np.ndarray | None,
        tol: float) -> np.ndarray:
    """Return for each segment from ``origins`` to ``targets`` whether a
    triangle other than ``target_ids`` and ``own_ids`` lies on it."""
    d = targets - origins
    length = np.linalg.norm(d, axis=1)
    direction = d / length[:, np.newaxis]

    # first triangle hit by each ray
    tri, ray, location = mesh.ray.intersects_id(
        origins, direction, multiple_hits=False, return_locations=True)
    distance = np.einsum(
        "ij,ij->i", location - origins[ray], direction[ray])
    blocking = (tri != target_ids[ray]) & (distance < length[ray] - tol)
    if own_ids is not None:
        blocking &= tri != own_ids[ray]

    blocked = np.zeros(origins.shape[0], dtype=bool)
    blocked[ray[blocking]] = True
    return blocked


def patch_visibility(
        patches_points: np.ndarray, patches_normal: np.ndarray,
        eps: float = 1e-5) -> np.ndarray:
    """Return the visibility between all triangular patches.

    Parameters
    ----------
    patches_points : np.ndarray
        vertices of the triangular patches of shape (n_patches, 3, 3)
    patches_normal : np.ndarray
        normals of the patches pointing into the room, of shape
        (n_patches, 3)
    eps : float, optional
        tolerance relative to the size of the room, by default 1e-5. It
        must stay well above the single precision of embree.

    Returns
    -------
    visibility_matrix : np.ndarray
        boolean matrix of shape (n_patches, n_patches). Only the upper
        triangle (``i < j``) is filled, as expected by sparrowpy.
    """
    patches_points = np.asarray(patches_points, dtype=float)
    patches_normal = np.asarray(patches_normal, dtype=float)
    n_patches = patches_points.shape[0]
    visibility = np.zeros((n_patches, n_patches), dtype=bool)
    if n_patches < 2:
        return visibility

    mesh = _room_mesh(patches_points)
    tol = eps * np.linalg.norm(mesh.extents)
    centers = patches_points.mean(axis=1)

    i_all, j_all = np.triu_indices(n_patches, 1)
    for start in range(0, i_all.size, _CHUNK_SIZE):
        i = i_all[start:start + _CHUNK_SIZE]
        j = j_all[start:start + _CHUNK_SIZE]
        d = centers[j] - centers[i]

        facing = (
            (np.einsum("ij,ij->i", patches_normal[i], d) > _PLANE_TOLERANCE) &
            (np.einsum("ij,ij->i", patches_normal[j], -d) > _PLANE_TOLERANCE))
        i, j = i[facing], j[facing]
        if i.size == 0:
            continue

        # start slightly in front of the source patch, so the ray does not
        # hit its own triangle (embree works in single precision)
        origins = centers[i] + patches_normal[i] * tol
        visible = ~_blocked(mesh, origins, centers[j], j, i, tol)
        visibility[i[visible], j[visible]] = True

    return visibility


def point_visibility(
        points: np.ndarray, patches_points: np.ndarray,
        patches_normal: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Return the visibility between points and all triangular patches.

    Parameters
    ----------
    points : np.ndarray
        positions of the sources or receivers of shape (n_points, 3)
    patches_points : np.ndarray
        vertices of the triangular patches of shape (n_patches, 3, 3)
    patches_normal : np.ndarray
        normals of the patches pointing into the room, of shape
        (n_patches, 3)
    eps : float, optional
        tolerance relative to the size of the room, by default 1e-5.

    Returns
    -------
    visibility : np.ndarray
        boolean matrix of shape (n_points, n_patches), True if the patch
        is visible from the point.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    patches_points = np.asarray(patches_points, dtype=float)
    patches_normal = np.asarray(patches_normal, dtype=float)
    n_points, n_patches = points.shape[0], patches_points.shape[0]

    mesh = _room_mesh(patches_points)
    tol = eps * np.linalg.norm(mesh.extents)
    centers = patches_points.mean(axis=1)

    p_all, j_all = np.divmod(np.arange(n_points * n_patches), n_patches)
    visibility = np.zeros((n_points, n_patches), dtype=bool)
    for start in range(0, p_all.size, _CHUNK_SIZE):
        p = p_all[start:start + _CHUNK_SIZE]
        j = j_all[start:start + _CHUNK_SIZE]

        # the point lies in front of the patch
        in_front = np.einsum(
            "ij,ij->i", patches_normal[j],
            points[p] - centers[j]) > _PLANE_TOLERANCE
        p, j = p[in_front], j[in_front]
        if p.size == 0:
            continue

        visible = ~_blocked(mesh, points[p], centers[j], j, None, tol)
        visibility[p[visible], j[visible]] = True

    return visibility
