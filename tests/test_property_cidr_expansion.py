# Feature: v2-attack-tracking-overhaul, Property 2: CIDR Expansion Produces Valid IPs
"""Property-based tests for CIDR expansion producing valid IPs.

Validates: Requirements 3.2

For any valid CIDR range with prefix length >= 24 (e.g., /24, /25, /26, /27, /28),
the expansion function SHALL produce between 5 and 10 IPs, all of which belong to
the specified network range.
"""

import ipaddress

from hypothesis import given, settings
from hypothesis import strategies as st

from services.threat_aggregator import expand_cidr


# --- Strategies ---

# Generate random first 3 octets + prefix lengths from 24 to 30
valid_cidr_st = st.builds(
    lambda o1, o2, o3, o4, prefix: f"{o1}.{o2}.{o3}.{o4}/{prefix}",
    o1=st.integers(min_value=1, max_value=254),
    o2=st.integers(min_value=0, max_value=255),
    o3=st.integers(min_value=0, max_value=255),
    o4=st.integers(min_value=0, max_value=255),
    prefix=st.integers(min_value=24, max_value=30),
)

# Generate CIDR with prefix < 24 (should return empty)
large_cidr_st = st.builds(
    lambda o1, o2, o3, o4, prefix: f"{o1}.{o2}.{o3}.{o4}/{prefix}",
    o1=st.integers(min_value=1, max_value=254),
    o2=st.integers(min_value=0, max_value=255),
    o3=st.integers(min_value=0, max_value=255),
    o4=st.integers(min_value=0, max_value=255),
    prefix=st.integers(min_value=1, max_value=23),
)

# Generate invalid CIDR strings
invalid_cidr_st = st.one_of(
    st.text(min_size=0, max_size=20),  # random text
    st.just("999.999.999.999/24"),  # invalid octets
    st.just("not_a_cidr"),
    st.just(""),
    st.just("/24"),
    st.just("192.168.1.0/"),
    st.just("192.168.1.0/33"),  # prefix too large
)


# --- Property 2: CIDR Expansion Produces Valid IPs ---


@settings(max_examples=200, deadline=None)
@given(cidr=valid_cidr_st)
def test_cidr_expansion_returns_5_to_10_ips_or_all_hosts(cidr: str) -> None:
    """For any valid IPv4 network with prefix >= 24, expand_cidr() returns
    between 5 and 10 IPs, or all hosts if fewer than 5 are available.

    **Validates: Requirements 3.2**
    """
    result = expand_cidr(cidr)
    network = ipaddress.IPv4Network(cidr, strict=False)
    total_hosts = len(list(network.hosts()))

    if total_hosts <= 5:
        # If the network has fewer hosts than the minimum sample size,
        # all hosts should be returned
        assert len(result) == total_hosts, (
            f"CIDR {cidr} has {total_hosts} hosts but expand_cidr returned "
            f"{len(result)} IPs"
        )
    else:
        assert 5 <= len(result) <= 10, (
            f"CIDR {cidr} should produce 5-10 IPs, got {len(result)}"
        )


@settings(max_examples=200, deadline=None)
@given(cidr=valid_cidr_st)
def test_cidr_expansion_all_ips_belong_to_network(cidr: str) -> None:
    """Every returned IP is a valid member of the specified network.

    **Validates: Requirements 3.2**
    """
    result = expand_cidr(cidr)
    network = ipaddress.IPv4Network(cidr, strict=False)

    for ip_str in result:
        ip = ipaddress.IPv4Address(ip_str)
        assert ip in network, (
            f"IP {ip_str} from expand_cidr({cidr!r}) is not in network {network}"
        )


@settings(max_examples=200, deadline=None)
@given(cidr=large_cidr_st)
def test_cidr_expansion_prefix_less_than_24_returns_empty(cidr: str) -> None:
    """For prefix < 24, expand_cidr() returns an empty list.

    **Validates: Requirements 3.2**
    """
    result = expand_cidr(cidr)
    assert result == [], (
        f"CIDR {cidr} with prefix < 24 should return empty list, got {result}"
    )


@settings(max_examples=100, deadline=None)
@given(cidr=invalid_cidr_st)
def test_cidr_expansion_invalid_input_returns_empty(cidr: str) -> None:
    """Invalid CIDR strings return an empty list.

    **Validates: Requirements 3.2**
    """
    result = expand_cidr(cidr)
    assert result == [], (
        f"Invalid CIDR {cidr!r} should return empty list, got {result}"
    )
