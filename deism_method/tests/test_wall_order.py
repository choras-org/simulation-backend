"""Regression tests for wall-surface ordering.

CHORAS rooms take DEISM's convex/ARG path, which re-matches each wall to its
absorption value by centroid proximity. ``get_deism_surface_order`` must
therefore produce an order that depends only on geometry (wall centroids), not
on the physical-tag declaration order of the mesh -- otherwise a change in
``.geo`` surface declaration order could silently permute boundary conditions.
"""

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


def _names(pairs):
    """Surface names from ``(tag, name)`` pairs, in declaration order."""
    return [name for _, name in pairs]


def test_order_is_independent_of_declaration_and_tags():
    """Two different tag/declaration orderings yield the identical wall order."""
    order_a = _names(
        [(2, "uuid-A"), (5, "uuid-B"), (4, "uuid-C"),
         (6, "uuid-D"), (1, "uuid-E"), (3, "uuid-F")]
    )
    # Different declaration order AND different tag->name assignment.
    order_b = _names(
        [(1, "uuid-F"), (2, "uuid-E"), (3, "uuid-D"),
         (4, "uuid-C"), (5, "uuid-B"), (6, "uuid-A")]
    )

    result_a = get_deism_surface_order(order_a, WALL_CENTERS)
    result_b = get_deism_surface_order(order_b, WALL_CENTERS)

    assert result_a == result_b
    # All six surfaces present exactly once -- no dropped or duplicated walls.
    assert sorted(result_a) == sorted(WALL_CENTERS)


def test_accepts_any_iterable_of_names():
    """The keys view of the JSON ``wall_centers`` mapping is accepted as-is."""
    assert get_deism_surface_order(WALL_CENTERS.keys(), WALL_CENTERS) == \
        get_deism_surface_order(list(WALL_CENTERS), WALL_CENTERS)


def test_fewer_than_four_surfaces_raises():
    """A room that cannot be a closed polyhedron (< 4 faces) is rejected."""
    three = _names([(2, "uuid-A"), (5, "uuid-B"), (4, "uuid-C")])
    with pytest.raises(ValueError):
        get_deism_surface_order(three, WALL_CENTERS)


def test_non_hexahedral_convex_rooms_accepted():
    """Convex rooms need not have 6 walls.

    DEISM's convex/ARG path accepts any wall count M >= 4; the wrapper must
    not impose a shoebox-like 6-wall requirement.
    """
    # 5 walls (e.g. a wedge) and 8 walls (e.g. a truncated box).
    for extra in ([], [("g", "uuid-G"), ("h", "uuid-H"), ("i", "uuid-I")]):
        pairs = [(2, "uuid-A"), (5, "uuid-B"), (4, "uuid-C"),
                 (6, "uuid-D"), (1, "uuid-E")] + extra
        centers = dict(WALL_CENTERS)
        centers.update({
            "uuid-G": [1.0, 0.5, 3.0],
            "uuid-H": [3.0, 2.5, 3.0],
            "uuid-I": [2.0, 2.9, 2.9],
        })
        result = get_deism_surface_order(_names(pairs), centers)
        assert sorted(result) == sorted(name for _, name in pairs)
