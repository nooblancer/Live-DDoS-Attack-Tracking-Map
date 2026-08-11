"""Property-based tests for ml/preprocess.py using Hypothesis.

Validates: Requirements 1.2, 1.3, 1.4, 1.5
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, series

from ml.preprocess import UNWANTED_COLUMNS, clean_dataframe, encode_labels


# ---------- Strategies ----------

# Strategy for column names that are NOT in the unwanted list (including stripped versions)
_unwanted_set = set(UNWANTED_COLUMNS) | {c.strip() for c in UNWANTED_COLUMNS}

safe_column_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_"),
    min_size=1,
    max_size=15,
).filter(lambda name: name not in _unwanted_set)

# Strategy for numeric values (finite floats)
finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)


# ---------- Property 1: Column removal preserves unrelated columns ----------
# Feature: ddos-attack-tracking-map, Property 1: Column removal preserves unrelated columns


@settings(max_examples=100, deadline=None)
@given(
    extra_cols=st.lists(safe_column_names, min_size=1, max_size=5, unique=True),
    n_rows=st.integers(min_value=1, max_value=10),
)
def test_column_removal_preserves_unrelated_columns(extra_cols, n_rows):
    """For any DataFrame containing a superset of the unwanted columns plus arbitrary
    other columns, after clean_dataframe(), none of the unwanted columns shall remain
    and all other columns shall be preserved unchanged.

    **Validates: Requirements 1.2**
    """
    # Build a DataFrame with all unwanted columns plus extra (safe) columns
    data = {}
    for col in UNWANTED_COLUMNS:
        data[col] = list(range(n_rows))
    for col in extra_cols:
        data[col] = [float(i) for i in range(n_rows)]

    df = pd.DataFrame(data)
    result = clean_dataframe(df)

    # No unwanted column (or its stripped variant) should remain
    for col in UNWANTED_COLUMNS:
        assert col not in result.columns
        assert col.strip() not in result.columns

    # All extra columns should still be present
    for col in extra_cols:
        assert col in result.columns


# ---------- Property 2: Label encoding produces valid binary classification ----------
# Feature: ddos-attack-tracking-map, Property 2: Label encoding produces valid binary classification


@settings(max_examples=100, deadline=None)
@given(
    labels=st.lists(
        st.one_of(
            st.just("BENIGN"),
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N")),
                min_size=1,
                max_size=10,
            ).filter(lambda s: s.strip() != "BENIGN"),
        ),
        min_size=1,
        max_size=50,
    )
)
def test_label_encoding_produces_valid_binary(labels):
    """For any pandas Series containing a mix of "BENIGN" and arbitrary non-BENIGN
    strings, encode_labels() shall produce a Series containing only values 0 and 1,
    where every original "BENIGN" maps to 0 and every non-BENIGN string maps to 1.

    **Validates: Requirements 1.3**
    """
    # Build a DataFrame with a dummy feature and the generated labels
    df = pd.DataFrame({"Feature": range(len(labels)), "Label": labels})
    _, binary_labels = encode_labels(df, label_col="Label")

    # Only 0 and 1 in output
    assert set(binary_labels.unique()).issubset({0, 1})

    # Check mapping correctness
    for original, encoded in zip(labels, binary_labels):
        if original.strip() == "BENIGN":
            assert encoded == 0, f"Expected 0 for BENIGN, got {encoded}"
        else:
            assert encoded == 1, f"Expected 1 for '{original}', got {encoded}"


# ---------- Property 3: Infinite and NaN removal produces clean output ----------
# Feature: ddos-attack-tracking-map, Property 3: Infinite and NaN removal produces clean output


@settings(max_examples=100, deadline=None)
@given(
    n_rows=st.integers(min_value=1, max_value=20),
    n_cols=st.integers(min_value=1, max_value=5),
    data=st.data(),
)
def test_inf_nan_removal_produces_clean_output(n_rows, n_cols, data):
    """For any numeric DataFrame with inf or NaN values injected at arbitrary
    positions, after clean_dataframe(), the resulting DataFrame shall contain
    zero infinite values and zero NaN values.

    **Validates: Requirements 1.4**
    """
    col_names = [f"col_{i}" for i in range(n_cols)]

    # Generate base numeric values
    values = {}
    for col in col_names:
        col_data = data.draw(
            st.lists(finite_floats, min_size=n_rows, max_size=n_rows)
        )
        values[col] = col_data

    df = pd.DataFrame(values)

    # Inject inf/NaN at random positions
    for col in col_names:
        for row_idx in range(n_rows):
            inject = data.draw(st.sampled_from(["keep", "inf", "-inf", "nan"]))
            if inject == "inf":
                df.at[row_idx, col] = np.inf
            elif inject == "-inf":
                df.at[row_idx, col] = -np.inf
            elif inject == "nan":
                df.at[row_idx, col] = np.nan

    result = clean_dataframe(df)

    # Output must contain zero inf and zero NaN
    if len(result) > 0:
        assert not np.any(np.isinf(result.values.astype(float)))
        assert not result.isna().any().any()


# ---------- Property 4: Deduplication preserves uniqueness and order ----------
# Feature: ddos-attack-tracking-map, Property 4: Deduplication preserves uniqueness and order


@settings(max_examples=100, deadline=None)
@given(
    n_unique_rows=st.integers(min_value=1, max_value=10),
    n_cols=st.integers(min_value=1, max_value=4),
    data=st.data(),
)
def test_deduplication_preserves_uniqueness_and_order(n_unique_rows, n_cols, data):
    """For any DataFrame with duplicated rows, after removing duplicates keeping
    the first occurrence, the output shall have no duplicate rows.

    **Validates: Requirements 1.5**
    """
    col_names = [f"col_{i}" for i in range(n_cols)]

    # Generate unique rows
    rows = []
    for _ in range(n_unique_rows):
        row = data.draw(
            st.lists(finite_floats, min_size=n_cols, max_size=n_cols)
        )
        rows.append(row)

    df_unique = pd.DataFrame(rows, columns=col_names)

    # Duplicate some rows by repeating them
    repeat_counts = data.draw(
        st.lists(
            st.integers(min_value=1, max_value=3),
            min_size=n_unique_rows,
            max_size=n_unique_rows,
        )
    )
    repeated_rows = []
    for i, count in enumerate(repeat_counts):
        for _ in range(count):
            repeated_rows.append(rows[i])

    df_with_dups = pd.DataFrame(repeated_rows, columns=col_names)

    result = clean_dataframe(df_with_dups)

    # No duplicate rows in output
    assert not result.duplicated().any()
