"""Regression tests for wall-surface ordering (issues.md #2).

CHORAS rooms take DEISM's convex/ARG path, which re-matches each wall to its
absorption value by centroid proximity. ``get_deism_surface_order`` must
therefore produce an order that depends only on geometry (wall centroids), not
on the Gmsh physical-tag declaration order -- otherwise a change in ``.geo``
surface declaration order could silently permute boundary conditions.
"""

import numpy as np
import pytest

from deism_interface.DEISMinterface import get_deism_surface_order


# Six walls of an arbitrary convex room, each with a distinct centroid.
WALL_CENTERS = {
    "uuid-A": [0.0, 1.5, 1.5],
    "uuid-B": [4.0, 1.5, 1.5],
    "uuid-C": [2.0, 0.0, 1.75],
    "uuid-D": [2.0, 3.0, 1.25],
    "uuid-E": [2.0, 1.5, 0.0],
    "uuid-F": [2.0, 1.5, 3.0],
}


def _vgroups(pairs):
    """Build ``[dim, tag, name]`` physical-group entries for dim=2 surfaces."""
    return [[2, tag, name] for tag, name in pairs]


def test_order_is_independent_of_declaration_and_tags():
    """Two different tag/declaration orderings yield the identical wall order."""
    order_a = _vgroups(
        [(2, "uuid-A"), (5, "uuid-B"), (4, "uuid-C"),
         (6, "uuid-D"), (1, "uuid-E"), (3, "uuid-F")]
    )
    # Different declaration order AND different tag->name assignment.
    order_b = _vgroups(
        [(1, "uuid-F"), (2, "uuid-E"), (3, "uuid-D"),
         (4, "uuid-C"), (5, "uuid-B"), (6, "uuid-A")]
    )

    result_a = get_deism_surface_order(order_a, WALL_CENTERS)
    result_b = get_deism_surface_order(order_b, WALL_CENTERS)

    assert result_a == result_b
    # All six surfaces present exactly once -- no dropped or duplicated walls.
    assert sorted(result_a) == sorted(WALL_CENTERS)


def test_ignores_non_surface_groups():
    """Volume/line physical groups (dim != 2) must not affect the result."""
    surfaces = _vgroups(
        [(2, "uuid-A"), (5, "uuid-B"), (4, "uuid-C"),
         (6, "uuid-D"), (1, "uuid-E"), (3, "uuid-F")]
    )
    with_extras = surfaces + [[3, 1, "RoomVolume"], [1, 1, "default"]]

    assert get_deism_surface_order(with_extras, WALL_CENTERS) == \
        get_deism_surface_order(surfaces, WALL_CENTERS)


def test_wrong_surface_count_raises():
    """A room without exactly six wall surfaces is rejected loudly."""
    five = _vgroups(
        [(2, "uuid-A"), (5, "uuid-B"), (4, "uuid-C"),
         (6, "uuid-D"), (1, "uuid-E")]
    )
    with pytest.raises(ValueError):
        get_deism_surface_order(five, WALL_CENTERS)
