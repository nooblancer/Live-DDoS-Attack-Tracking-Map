# Feature: v2-attack-tracking-overhaul, Property 9: Geographic Coordinate Pool Membership
"""Property-based tests for geographic coordinate pool membership.

Validates: Requirements 1.3, 13.1

For any flow processed by the Replay Engine, the assigned source coordinates
(latitude, longitude) SHALL belong to one of the defined geographic pools, where
each pool represents a valid country region with coordinates within that country's
geographic bounds.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from models.schemas import AttackType
from services.geo_pools import COUNTRY_POOLS, GeoCoordinatePools


# --- Helpers ---

# Build a flat set of all valid (lat, lon) tuples across all pools
ALL_POOL_COORDINATES: set[tuple[float, float]] = set()
for _coords_list in COUNTRY_POOLS.values():
    for _coord in _coords_list:
        ALL_POOL_COORDINATES.add(_coord)

# All attack type values as strings (matching ATTACK_TYPE_REGION_BIAS keys)
ATTACK_TYPE_VALUES: list[str] = [at.value for at in AttackType]


# --- Strategies ---

attack_type_st = st.sampled_from(ATTACK_TYPE_VALUES)
optional_attack_type_st = st.one_of(st.none(), attack_type_st)


# --- Property 9: Geographic Coordinate Pool Membership ---


@settings(max_examples=200, deadline=None)
@given(attack_type=attack_type_st)
def test_source_coords_with_attack_type_belong_to_pool(attack_type: str) -> None:
    """For each of the 12 attack types, get_source_coords(attack_type) returns
    coordinates from one of the COUNTRY_POOLS lists.

    **Validates: Requirements 1.3, 13.1**
    """
    pools = GeoCoordinatePools()
    lat, lon = pools.get_source_coords(attack_type=attack_type)

    assert (lat, lon) in ALL_POOL_COORDINATES, (
        f"Coordinates ({lat}, {lon}) for attack_type={attack_type!r} "
        f"not found in any COUNTRY_POOLS entry"
    )


@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_source_coords_without_attack_type_belong_to_pool(data: st.DataObject) -> None:
    """With no attack_type (None), coordinates are still from a valid pool.

    **Validates: Requirements 1.3, 13.1**
    """
    pools = GeoCoordinatePools()
    lat, lon = pools.get_source_coords(attack_type=None)

    assert (lat, lon) in ALL_POOL_COORDINATES, (
        f"Coordinates ({lat}, {lon}) with attack_type=None "
        f"not found in any COUNTRY_POOLS entry"
    )


@settings(max_examples=200, deadline=None)
@given(attack_type=optional_attack_type_st)
def test_source_coords_any_call_belongs_to_pool(attack_type: str | None) -> None:
    """For any call to get_source_coords() (with or without attack_type), the
    returned (lat, lon) tuple exists in one of the COUNTRY_POOLS lists.

    **Validates: Requirements 1.3, 13.1**
    """
    pools = GeoCoordinatePools()
    lat, lon = pools.get_source_coords(attack_type=attack_type)

    assert (lat, lon) in ALL_POOL_COORDINATES, (
        f"Coordinates ({lat}, {lon}) for attack_type={attack_type!r} "
        f"not found in any COUNTRY_POOLS entry"
    )


def test_country_pools_covers_at_least_30_countries() -> None:
    """The pool covers at least 30 countries.

    **Validates: Requirements 1.3, 13.1**
    """
    assert len(COUNTRY_POOLS) >= 30, (
        f"COUNTRY_POOLS contains only {len(COUNTRY_POOLS)} countries, "
        f"expected at least 30"
    )


def test_all_pool_coordinates_are_valid_lat_lon() -> None:
    """All coordinates in COUNTRY_POOLS are valid lat/lon ranges
    (-90 to 90, -180 to 180).

    **Validates: Requirements 1.3, 13.1**
    """
    for country, coords_list in COUNTRY_POOLS.items():
        for lat, lon in coords_list:
            assert -90.0 <= lat <= 90.0, (
                f"{country}: latitude {lat} is out of range [-90, 90]"
            )
            assert -180.0 <= lon <= 180.0, (
                f"{country}: longitude {lon} is out of range [-180, 180]"
            )
