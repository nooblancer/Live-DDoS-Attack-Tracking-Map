# Feature: ddos-attack-tracking-map, Property 12: Geolocation batch sizes respect configured limit
# Feature: ddos-attack-tracking-map, Property 13: Geolocation response parsing extracts successful entries only
"""Property-based tests for geolocation batching and response parsing.

Validates: Requirements 6.1, 6.3, 6.4
"""

import asyncio
import json
import math

import httpx
from hypothesis import given, settings
from hypothesis import strategies as st

from services.geolocation import GeolocationService, IP_API_BATCH_URL


# --- Strategies ---

# Generate valid IPv4 addresses
ipv4_st = st.tuples(
    st.integers(min_value=1, max_value=223),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=1, max_value=254),
).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}")

# Generate a list of IPs (1-50)
ip_list_st = st.lists(ipv4_st, min_size=1, max_size=50)

# Generate batch sizes (1-20)
batch_size_st = st.integers(min_value=1, max_value=20)

# Generate a successful geolocation entry
success_entry_st = st.fixed_dictionaries(
    {
        "status": st.just("success"),
        "query": ipv4_st,
        "lat": st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        "lon": st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        "country": st.text(min_size=1, max_size=30, alphabet=st.characters(categories=("L", "Zs"))),
        "city": st.text(min_size=1, max_size=30, alphabet=st.characters(categories=("L", "Zs"))),
        "isp": st.text(min_size=1, max_size=50, alphabet=st.characters(categories=("L", "N", "Zs"))),
    }
)

# Generate a failed geolocation entry
fail_entry_st = st.fixed_dictionaries(
    {
        "status": st.just("fail"),
        "query": ipv4_st,
        "message": st.sampled_from(["private range", "reserved range", "invalid query"]),
    }
)

# Generate a mixed response with at least one success and at least one fail
mixed_response_st = st.tuples(
    st.lists(success_entry_st, min_size=1, max_size=10),
    st.lists(fail_entry_st, min_size=1, max_size=10),
).map(lambda t: t[0] + t[1])


# --- Property 12: Geolocation batch sizes respect configured limit ---


@settings(max_examples=100, deadline=None)
@given(ips=ip_list_st, batch_size=batch_size_st)
def test_batch_sizes_respect_configured_limit(
    ips: list[str], batch_size: int
) -> None:
    """For any list of N IP addresses and a configured GEO_BATCH_SIZE of B,
    all batch requests sent to ip-api.com shall contain at most B IP addresses each.

    **Validates: Requirements 6.1**
    """

    async def _run():
        # Track all request bodies sent
        captured_bodies: list[list[str]] = []

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            captured_bodies.append(body)
            # Return minimal success response
            response_data = [
                {
                    "status": "success",
                    "query": ip,
                    "lat": 0.0,
                    "lon": 0.0,
                    "country": "Test",
                    "city": "Test",
                    "isp": "Test",
                }
                for ip in body
            ]
            return httpx.Response(200, json=response_data)

        service = GeolocationService(batch_size=batch_size)
        service._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        try:
            await service.geolocate_batch(ips)

            # Verify expected number of batches
            expected_batches = math.ceil(len(ips) / batch_size)
            assert len(captured_bodies) == expected_batches, (
                f"Expected {expected_batches} batch requests, got {len(captured_bodies)}"
            )

            # Verify each batch respects the size limit
            for i, body in enumerate(captured_bodies):
                assert len(body) <= batch_size, (
                    f"Batch {i} contained {len(body)} IPs, "
                    f"exceeds configured limit of {batch_size}"
                )
                # Batches should not be empty
                assert len(body) >= 1, f"Batch {i} was empty"
        finally:
            await service._client.aclose()
            service._client = None

    asyncio.run(_run())


# --- Property 13: Geolocation response parsing extracts successful entries only ---


@settings(max_examples=100, deadline=None)
@given(response_entries=mixed_response_st)
def test_response_parsing_extracts_successful_entries_only(
    response_entries: list[dict],
) -> None:
    """For any ip-api.com batch response containing a mix of "success" and "fail"
    status entries, the parsed output shall contain exactly those entries with
    "success" status, each with latitude, longitude, country, city, and ISP extracted.

    **Validates: Requirements 6.3, 6.4**
    """

    async def _run():
        # Extract the IPs from our generated response to use as input
        ips = [entry["query"] for entry in response_entries]

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            # Return our pre-generated mixed response
            return httpx.Response(200, json=response_entries)

        # Use a batch size large enough to fit all IPs in one request
        service = GeolocationService(batch_size=len(ips) + 1)
        service._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        try:
            results = await service.geolocate_batch(ips)

            # Determine expected successful entries
            expected_success = [e for e in response_entries if e["status"] == "success"]

            # Result count must match the number of success entries
            assert len(results) == len(expected_success), (
                f"Expected {len(expected_success)} results, got {len(results)}"
            )

            # Each result must have the correct fields extracted
            for result, expected in zip(results, expected_success):
                assert result["ip"] == expected["query"], (
                    f"IP mismatch: {result['ip']} != {expected['query']}"
                )
                assert result["lat"] == expected["lat"], (
                    f"Latitude mismatch for {result['ip']}"
                )
                assert result["lon"] == expected["lon"], (
                    f"Longitude mismatch for {result['ip']}"
                )
                assert result["country"] == expected["country"], (
                    f"Country mismatch for {result['ip']}"
                )
                assert result["city"] == expected["city"], (
                    f"City mismatch for {result['ip']}"
                )
                assert result["isp"] == expected["isp"], (
                    f"ISP mismatch for {result['ip']}"
                )
        finally:
            await service._client.aclose()
            service._client = None

    asyncio.run(_run())
