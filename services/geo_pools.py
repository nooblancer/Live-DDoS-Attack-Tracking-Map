"""Geographic coordinate pools for assigning realistic source locations to replay flows.

Provides weighted random selection from known botnet-origin regions across 30+
countries on all continents, creating visually diverse attack trajectories on the
3D globe.
"""

import os
import random
import logging

logger = logging.getLogger(__name__)

# Default target coordinate: Ashburn, Virginia (major US data center hub)
_DEFAULT_TARGET_LAT = 39.0438
_DEFAULT_TARGET_LON = -77.4874

# Coordinate pools per country: list of (lat, lon) tuples representing major cities/regions
# Each country has 3-5 coordinate pairs for visual diversity
COUNTRY_POOLS: dict[str, list[tuple[float, float]]] = {
    # --- High botnet activity regions (heavily weighted) ---
    "Russia": [
        (55.7558, 37.6173),   # Moscow
        (59.9343, 30.3351),   # St. Petersburg
        (56.8389, 60.6057),   # Yekaterinburg
        (55.0084, 82.9357),   # Novosibirsk
        (43.1332, 131.9113),  # Vladivostok
    ],
    "China": [
        (39.9042, 116.4074),  # Beijing
        (31.2304, 121.4737),  # Shanghai
        (23.1291, 113.2644),  # Guangzhou
        (22.5431, 114.0579),  # Shenzhen
        (30.5728, 104.0668),  # Chengdu
    ],
    "Brazil": [
        (-23.5505, -46.6333),  # São Paulo
        (-22.9068, -43.1729),  # Rio de Janeiro
        (-15.7975, -47.8919),  # Brasília
        (-3.1190, -60.0217),   # Manaus
        (-12.9714, -38.5124),  # Salvador
    ],
    "Vietnam": [
        (21.0278, 105.8342),  # Hanoi
        (10.8231, 106.6297),  # Ho Chi Minh City
        (16.0544, 108.2022),  # Da Nang
        (10.0452, 105.7469),  # Can Tho
        (21.1924, 105.9740),  # Bac Ninh
    ],
    "Indonesia": [
        (-6.2088, 106.8456),   # Jakarta
        (-7.2575, 112.7521),   # Surabaya
        (-6.9175, 107.6191),   # Bandung
        (-8.6705, 115.2126),   # Bali/Denpasar
        (3.5952, 98.6722),     # Medan
    ],
    "Turkey": [
        (41.0082, 28.9784),   # Istanbul
        (39.9334, 32.8597),   # Ankara
        (38.4192, 27.1287),   # Izmir
        (37.0000, 35.3213),   # Adana
        (40.1885, 29.0610),   # Bursa
    ],
    "India": [
        (19.0760, 72.8777),   # Mumbai
        (28.6139, 77.2090),   # New Delhi
        (12.9716, 77.5946),   # Bangalore
        (13.0827, 80.2707),   # Chennai
        (22.5726, 88.3639),   # Kolkata
    ],
    "Ukraine": [
        (50.4501, 30.5234),   # Kyiv
        (49.9935, 36.2304),   # Kharkiv
        (46.4825, 30.7233),   # Odesa
        (48.4647, 35.0462),   # Dnipro
    ],
    "Iran": [
        (35.6892, 51.3890),   # Tehran
        (32.6546, 51.6680),   # Isfahan
        (29.5918, 52.5837),   # Shiraz
        (36.2605, 59.6168),   # Mashhad
    ],
    "Pakistan": [
        (24.8607, 67.0011),   # Karachi
        (31.5204, 74.3587),   # Lahore
        (33.6844, 73.0479),   # Islamabad
        (25.3960, 68.3578),   # Hyderabad
    ],
    # --- Moderate botnet activity ---
    "Nigeria": [
        (6.5244, 3.3792),     # Lagos
        (9.0579, 7.4951),     # Abuja
        (7.3775, 3.9470),     # Ibadan
        (6.3350, 5.6037),     # Benin City
    ],
    "Bangladesh": [
        (23.8103, 90.4125),   # Dhaka
        (22.3569, 91.7832),   # Chittagong
        (24.3636, 88.6241),   # Rajshahi
    ],
    "Thailand": [
        (13.7563, 100.5018),  # Bangkok
        (18.7883, 98.9853),   # Chiang Mai
        (7.8804, 98.3923),    # Phuket
    ],
    "Philippines": [
        (14.5995, 120.9842),  # Manila
        (10.3157, 123.8854),  # Cebu
        (7.1907, 125.4553),   # Davao
    ],
    "South Korea": [
        (37.5665, 126.9780),  # Seoul
        (35.1796, 129.0756),  # Busan
        (35.1595, 126.8526),  # Gwangju
    ],
    "Taiwan": [
        (25.0330, 121.5654),  # Taipei
        (22.6273, 120.3014),  # Kaohsiung
        (24.1477, 120.6736),  # Taichung
    ],
    "Mexico": [
        (19.4326, -99.1332),  # Mexico City
        (20.6597, -103.3496), # Guadalajara
        (25.6866, -100.3161), # Monterrey
    ],
    "Argentina": [
        (-34.6037, -58.3816), # Buenos Aires
        (-31.4201, -64.1888), # Córdoba
        (-32.9468, -60.6393), # Rosario
    ],
    "Colombia": [
        (4.7110, -74.0721),   # Bogotá
        (6.2442, -75.5812),   # Medellín
        (3.4516, -76.5320),   # Cali
    ],
    "Egypt": [
        (30.0444, 31.2357),   # Cairo
        (31.2001, 29.9187),   # Alexandria
        (30.0131, 31.2089),   # Giza
    ],
    "South Africa": [
        (-33.9249, 18.4241),  # Cape Town
        (-26.2041, 28.0473),  # Johannesburg
        (-29.8587, 31.0218),  # Durban
    ],
    "Germany": [
        (52.5200, 13.4050),   # Berlin
        (48.1351, 11.5820),   # Munich
        (50.1109, 8.6821),    # Frankfurt
    ],
    "Netherlands": [
        (52.3676, 4.9041),    # Amsterdam
        (51.9244, 4.4777),    # Rotterdam
        (52.0907, 5.1214),    # Utrecht
    ],
    "Romania": [
        (44.4268, 26.1025),   # Bucharest
        (46.7712, 23.6236),   # Cluj-Napoca
        (45.7489, 21.2087),   # Timișoara
    ],
    "Poland": [
        (52.2297, 21.0122),   # Warsaw
        (50.0647, 19.9450),   # Kraków
        (51.7592, 19.4560),   # Łódź
    ],
    "United States": [
        (40.7128, -74.0060),  # New York
        (34.0522, -118.2437), # Los Angeles
        (41.8781, -87.6298),  # Chicago
        (29.7604, -95.3698),  # Houston
    ],
    "Japan": [
        (35.6762, 139.6503),  # Tokyo
        (34.6937, 135.5023),  # Osaka
        (35.0116, 135.7681),  # Kyoto
    ],
    "Malaysia": [
        (3.1390, 101.6869),   # Kuala Lumpur
        (5.4164, 100.3327),   # Penang
        (1.4927, 103.7414),   # Johor Bahru
    ],
    "Kenya": [
        (-1.2921, 36.8219),   # Nairobi
        (-4.0435, 39.6682),   # Mombasa
        (0.0917, 34.7680),    # Kisumu
    ],
    "Morocco": [
        (33.9716, -6.8498),   # Rabat
        (33.5731, -7.5898),   # Casablanca
        (31.6295, -7.9811),   # Marrakech
    ],
    "Australia": [
        (-33.8688, 151.2093), # Sydney
        (-37.8136, 144.9631), # Melbourne
        (-27.4698, 153.0251), # Brisbane
    ],
    "Saudi Arabia": [
        (24.7136, 46.6753),   # Riyadh
        (21.3891, 39.8579),   # Jeddah
        (26.4207, 50.0888),   # Dammam
    ],
}

# Weights for country selection — higher values = more likely to be selected
# Botnet-heavy regions get significantly higher weights
COUNTRY_WEIGHTS: dict[str, float] = {
    # Heavy botnet origins
    "Russia": 12.0,
    "China": 14.0,
    "Brazil": 8.0,
    "Vietnam": 7.0,
    "Indonesia": 7.0,
    "Turkey": 6.0,
    "India": 9.0,
    "Ukraine": 5.0,
    "Iran": 5.0,
    "Pakistan": 4.0,
    # Moderate
    "Nigeria": 4.0,
    "Bangladesh": 3.0,
    "Thailand": 3.0,
    "Philippines": 3.0,
    "South Korea": 2.5,
    "Taiwan": 2.0,
    "Mexico": 3.0,
    "Argentina": 2.0,
    "Colombia": 2.5,
    "Egypt": 3.0,
    "South Africa": 2.0,
    "Germany": 2.0,
    "Netherlands": 2.0,
    "Romania": 3.0,
    "Poland": 2.0,
    "United States": 3.0,
    "Japan": 1.5,
    "Malaysia": 2.0,
    "Kenya": 1.5,
    "Morocco": 1.5,
    "Australia": 1.0,
    "Saudi Arabia": 2.0,
}

# Optional per-attack-type bias: certain attack types may favor specific regions
# Maps attack_type → list of countries that are more likely sources for that type
ATTACK_TYPE_REGION_BIAS: dict[str, list[str]] = {
    "SYN Flood": ["China", "Russia", "Vietnam", "Indonesia"],
    "UDP Flood": ["Brazil", "India", "Turkey", "Pakistan"],
    "DNS Amplification": ["Russia", "China", "Ukraine", "Romania"],
    "HTTP Flood": ["China", "Vietnam", "Indonesia", "Thailand"],
    "LDAP": ["Russia", "Netherlands", "Germany", "United States"],
    "NTP": ["China", "Russia", "Brazil", "South Korea"],
    "MSSQL": ["China", "India", "Russia", "Turkey"],
    "NetBIOS": ["India", "Bangladesh", "Pakistan", "Vietnam"],
    "SSDP": ["Brazil", "Argentina", "Colombia", "Mexico"],
    "TFTP": ["Indonesia", "Vietnam", "Philippines", "Malaysia"],
    "UDPLag": ["Russia", "Ukraine", "Romania", "Poland"],
    "WebDDoS": ["China", "Vietnam", "Indonesia", "Thailand"],
}


class GeoCoordinatePools:
    """Provides random geographic coordinates from known botnet-origin regions.

    Selects source coordinates from weighted country pools covering 30+ countries
    across all continents. Botnet-heavy regions (Russia, China, Brazil, Vietnam,
    Indonesia, Turkey, India) are weighted more heavily for realistic distribution.
    """

    def __init__(self) -> None:
        self._countries = list(COUNTRY_POOLS.keys())
        self._weights = [COUNTRY_WEIGHTS.get(c, 1.0) for c in self._countries]

        # Load configurable target coordinates from environment
        target_lat = float(os.getenv("TARGET_LAT", str(_DEFAULT_TARGET_LAT)))
        target_lon = float(os.getenv("TARGET_LON", str(_DEFAULT_TARGET_LON)))
        self._target_coords = (target_lat, target_lon)

        logger.info(
            "GeoCoordinatePools initialized: %d countries, target=(%s, %s)",
            len(self._countries),
            target_lat,
            target_lon,
        )

    def get_source_coords(self, attack_type: str | None = None) -> tuple[float, float]:
        """Return (lat, lon) from a randomly selected botnet-origin pool.

        If attack_type is provided and has a regional bias mapping, there's a 60%
        chance the coordinates come from the biased region set. Otherwise (or for
        the remaining 40%), standard weighted selection is used.

        Args:
            attack_type: Optional attack type string to influence region selection.

        Returns:
            A (latitude, longitude) tuple from one of the geographic pools.
        """
        # Check if attack_type has regional bias and apply it probabilistically
        if attack_type and attack_type in ATTACK_TYPE_REGION_BIAS:
            if random.random() < 0.6:
                # Select from biased regions for this attack type
                biased_countries = ATTACK_TYPE_REGION_BIAS[attack_type]
                # Filter to countries that exist in our pools
                valid_biased = [c for c in biased_countries if c in COUNTRY_POOLS]
                if valid_biased:
                    country = random.choice(valid_biased)
                    return random.choice(COUNTRY_POOLS[country])

        # Standard weighted selection across all countries
        country = random.choices(self._countries, weights=self._weights, k=1)[0]
        return random.choice(COUNTRY_POOLS[country])

    @property
    def target_coords(self) -> tuple[float, float]:
        """Return the fixed target coordinate (defended network).

        Configurable via TARGET_LAT and TARGET_LON environment variables.
        Defaults to Ashburn, Virginia (39.0438, -77.4874) — a major US data center hub.
        """
        return self._target_coords
