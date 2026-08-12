"""Property-based tests for attack type color mapping completeness."""

# Feature: v2-attack-tracking-overhaul, Property 11: Attack Type Color Mapping Completeness

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from models.schemas import AttackType, ATTACK_TYPE_COLORS, ATTACK_TYPE_INDEX


# --- Strategies ---

# Strategy that draws from all valid AttackType enum values
attack_type_strategy = st.sampled_from(list(AttackType))

# Strategy that draws from all valid ATTACK_TYPE_INDEX keys (0-11)
attack_type_index_strategy = st.integers(min_value=0, max_value=11)

# Valid hex color pattern
HEX_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")


# --- Property Tests ---


@settings(max_examples=100)
@given(attack_type=attack_type_strategy)
def test_every_attack_type_has_color_mapping(attack_type: AttackType):
    """Property 11: Every AttackType enum value has a corresponding entry in ATTACK_TYPE_COLORS.

    For any valid attack type string (one of the 12 defined types), the color mapping
    SHALL return a defined hex color string.

    **Validates: Requirements 7.1, 8.4**
    """
    # The attack type's value must exist as a key in ATTACK_TYPE_COLORS
    assert attack_type.value in ATTACK_TYPE_COLORS, (
        f"AttackType '{attack_type.value}' has no entry in ATTACK_TYPE_COLORS"
    )


@settings(max_examples=100)
@given(attack_type=attack_type_strategy)
def test_color_mapping_returns_valid_hex(attack_type: AttackType):
    """Property 11: Every color returned is a valid hex color string (#XXXXXX pattern).

    For any valid attack type, the color mapping function SHALL return a defined
    hex color string from the specification.

    **Validates: Requirements 7.1, 8.4**
    """
    color = ATTACK_TYPE_COLORS[attack_type.value]
    assert HEX_COLOR_PATTERN.match(color), (
        f"Color '{color}' for attack type '{attack_type.value}' is not a valid hex color (#XXXXXX)"
    )


def test_color_mapping_covers_exactly_12_types():
    """Property 11: The color mapping covers exactly 12 types (no more, no less).

    The ATTACK_TYPE_COLORS dict SHALL have exactly 12 entries corresponding to
    the 12 defined attack types.

    **Validates: Requirements 7.1, 8.4**
    """
    # There must be exactly 12 attack types in the enum
    assert len(AttackType) == 12, (
        f"Expected 12 attack types in enum, got {len(AttackType)}"
    )

    # ATTACK_TYPE_COLORS must have exactly 12 entries
    assert len(ATTACK_TYPE_COLORS) == 12, (
        f"Expected 12 entries in ATTACK_TYPE_COLORS, got {len(ATTACK_TYPE_COLORS)}"
    )

    # Every enum value must be a key in ATTACK_TYPE_COLORS (completeness)
    enum_values = {at.value for at in AttackType}
    color_keys = set(ATTACK_TYPE_COLORS.keys())
    assert enum_values == color_keys, (
        f"Mismatch between AttackType values and ATTACK_TYPE_COLORS keys. "
        f"Missing in colors: {enum_values - color_keys}. "
        f"Extra in colors: {color_keys - enum_values}"
    )


@settings(max_examples=100)
@given(index=attack_type_index_strategy)
def test_attack_type_index_maps_all_indices_to_valid_labels(index: int):
    """Property 11: ATTACK_TYPE_INDEX maps all 12 indices (0-11) to valid attack type labels.

    For any index in [0, 11], ATTACK_TYPE_INDEX SHALL map to a label that is a valid
    AttackType value and also exists in ATTACK_TYPE_COLORS.

    **Validates: Requirements 7.1, 8.4**
    """
    # Index must exist in ATTACK_TYPE_INDEX
    assert index in ATTACK_TYPE_INDEX, (
        f"Index {index} not found in ATTACK_TYPE_INDEX"
    )

    label = ATTACK_TYPE_INDEX[index]

    # The label must be a valid AttackType value
    valid_labels = {at.value for at in AttackType}
    assert label in valid_labels, (
        f"ATTACK_TYPE_INDEX[{index}] = '{label}' is not a valid AttackType value"
    )

    # The label must also have a color mapping
    assert label in ATTACK_TYPE_COLORS, (
        f"ATTACK_TYPE_INDEX[{index}] = '{label}' has no color in ATTACK_TYPE_COLORS"
    )
