# Feature: v2-attack-tracking-overhaul, Property 10: Replay Target Coordinate Invariant
"""Property-based tests for replay target coordinate invariant.

For any replay event emitted by the Replay Engine, the target coordinates SHALL equal
the configured `target_coords` value exactly, regardless of attack type or source coordinates.

**Validates: Requirements 13.3**
"""

import os
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from services.geo_pools import GeoCoordinatePools, _DEFAULT_TARGET_LAT, _DEFAULT_TARGET_LON


# --- Strategies ---

# All 12 defined attack types from CIC-DDoS2019
ATTACK_TYPES = [
    "SYN Flood",
    "UDP Flood",
    "DNS Amplification",
    "HTTP Flood",
    "LDAP",
    "NTP",
    "MSSQL",
    "NetBIOS",
    "SSDP",
    "TFTP",
    "UDPLag",
    "WebDDoS",
]

attack_type_st = st.sampled_from(ATTACK_TYPES)

# Number of accesses to test immutability
access_count_st = st.integers(min_value=1, max_value=50)

# Number of get_source_coords calls to interleave
source_calls_st = st.integers(min_value=1, max_value=30)


# --- Property 10: Replay Target Coordinate Invariant ---


@settings(max_examples=100, deadline=None)
@given(access_count=access_count_st)
def test_target_coords_immutable_across_accesses(access_count: int) -> None:
    """The `target_coords` property always returns the same fixed tuple
    regardless of how many times it's accessed.

    **Validates: Requirements 13.3**
    """
    # Clear env vars to use defaults
    with patch.dict(os.environ, {}, clear=True):
        # Remove TARGET_LAT/TARGET_LON if present
        env = {k: v for k, v in os.environ.items() if k not in ("TARGET_LAT", "TARGET_LON")}
        with patch.dict(os.environ, env, clear=True):
            pool = GeoCoordinatePools()
            first_result = pool.target_coords

            for _ in range(access_count):
                current = pool.target_coords
                assert current == first_result, (
                    f"target_coords changed: expected {first_result}, got {current}"
                )


@settings(max_examples=100, deadline=None)
@given(
    attack_types=st.lists(attack_type_st, min_size=1, max_size=30),
)
def test_target_coords_unchanged_after_source_coord_calls(
    attack_types: list[str],
) -> None:
    """After calling `get_source_coords()` multiple times with different attack types,
    `target_coords` remains unchanged.

    **Validates: Requirements 13.3**
    """
    with patch.dict(os.environ, {}, clear=True):
        env = {k: v for k, v in os.environ.items() if k not in ("TARGET_LAT", "TARGET_LON")}
        with patch.dict(os.environ, env, clear=True):
            pool = GeoCoordinatePools()
            original_target = pool.target_coords

            # Call get_source_coords with various attack types
            for attack_type in attack_types:
                pool.get_source_coords(attack_type)

            # target_coords must still be the same
            assert pool.target_coords == original_target, (
                f"target_coords changed after get_source_coords calls: "
                f"expected {original_target}, got {pool.target_coords}"
            )


@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_target_coords_defaults_to_ashburn_virginia(data: st.DataObject) -> None:
    """The target coordinates default to Ashburn, Virginia (39.0438, -77.4874)
    when no env vars are set.

    **Validates: Requirements 13.3**
    """
    with patch.dict(os.environ, {}, clear=True):
        env = {k: v for k, v in os.environ.items() if k not in ("TARGET_LAT", "TARGET_LON")}
        with patch.dict(os.environ, env, clear=True):
            pool = GeoCoordinatePools()
            target = pool.target_coords

            assert target == (_DEFAULT_TARGET_LAT, _DEFAULT_TARGET_LON), (
                f"Expected default Ashburn coords ({_DEFAULT_TARGET_LAT}, {_DEFAULT_TARGET_LON}), "
                f"got {target}"
            )
            # Verify the actual default values
            assert target == (39.0438, -77.4874), (
                f"Default target not Ashburn, Virginia: got {target}"
            )


@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_target_coords_returns_tuple_of_two_floats(data: st.DataObject) -> None:
    """The target_coords property returns a tuple of exactly 2 floats (lat, lon).

    **Validates: Requirements 13.3**
    """
    with patch.dict(os.environ, {}, clear=True):
        env = {k: v for k, v in os.environ.items() if k not in ("TARGET_LAT", "TARGET_LON")}
        with patch.dict(os.environ, env, clear=True):
            pool = GeoCoordinatePools()
            target = pool.target_coords

            # Must be a tuple
            assert isinstance(target, tuple), (
                f"target_coords is not a tuple: type={type(target)}"
            )
            # Must have exactly 2 elements
            assert len(target) == 2, (
                f"target_coords has {len(target)} elements, expected 2"
            )
            # Both elements must be floats (or int, which is acceptable as numeric)
            lat, lon = target
            assert isinstance(lat, (int, float)), (
                f"Latitude is not a number: type={type(lat)}"
            )
            assert isinstance(lon, (int, float)), (
                f"Longitude is not a number: type={type(lon)}"
            )


@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_target_coords_within_valid_lat_lon_ranges(data: st.DataObject) -> None:
    """The target coordinates are within valid lat/lon ranges
    (lat: -90 to 90, lon: -180 to 180).

    **Validates: Requirements 13.3**
    """
    with patch.dict(os.environ, {}, clear=True):
        env = {k: v for k, v in os.environ.items() if k not in ("TARGET_LAT", "TARGET_LON")}
        with patch.dict(os.environ, env, clear=True):
            pool = GeoCoordinatePools()
            lat, lon = pool.target_coords

            assert -90.0 <= lat <= 90.0, (
                f"Latitude {lat} is outside valid range [-90, 90]"
            )
            assert -180.0 <= lon <= 180.0, (
                f"Longitude {lon} is outside valid range [-180, 180]"
            )
