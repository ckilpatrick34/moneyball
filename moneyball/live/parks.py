"""Home parks: Retrosheet site id (keys the park-factor state), location, roof.

roof: "open", "dome" (always closed) or "retract" (decided per game from the forecast).
Update when a team moves (the A's are in Sacramento through 2027; the Rays are
back at Tropicana Field from 2026).
"""

PARKS = {
    "ARI": ("PHO01", 33.445, -112.067, "retract"),
    "ATL": ("ATL03", 33.891, -84.468, "open"),
    "BAL": ("BAL12", 39.284, -76.622, "open"),
    "BOS": ("BOS07", 42.346, -71.097, "open"),
    "CHC": ("CHI11", 41.948, -87.656, "open"),
    "CWS": ("CHI12", 41.830, -87.634, "open"),
    "CIN": ("CIN09", 39.097, -84.507, "open"),
    "CLE": ("CLE08", 41.496, -81.685, "open"),
    "COL": ("DEN02", 39.756, -104.994, "open"),
    "DET": ("DET05", 42.339, -83.049, "open"),
    "HOU": ("HOU03", 29.757, -95.355, "dome"),     # roof closed ~99% of games
    "KC": ("KAN06", 39.051, -94.480, "open"),
    "LAA": ("ANA01", 33.800, -117.883, "open"),
    "LAD": ("LOS03", 34.074, -118.240, "open"),
    "MIA": ("MIA02", 25.778, -80.220, "retract"),
    "MIL": ("MIL06", 43.028, -87.971, "retract"),
    "MIN": ("MIN04", 44.982, -93.278, "open"),
    "NYM": ("NYC20", 40.757, -73.846, "open"),
    "NYY": ("NYC21", 40.829, -73.926, "open"),
    "OAK": ("SAC01", 38.580, -121.514, "open"),
    "PHI": ("PHI13", 39.906, -75.166, "open"),
    "PIT": ("PIT08", 40.447, -80.006, "open"),
    "SD": ("SAN02", 32.707, -117.157, "open"),
    "SEA": ("SEA03", 47.591, -122.332, "retract"),
    "SF": ("SFO03", 37.779, -122.389, "open"),
    "STL": ("STL10", 38.623, -90.193, "open"),
    "TB": ("STP01", 27.768, -82.653, "dome"),
    "TEX": ("ARL03", 32.747, -97.084, "retract"),
    "TOR": ("TOR02", 43.641, -79.389, "retract"),
    "WSH": ("WAS11", 38.873, -77.007, "open"),
}

# MLB Stats API / Odds API spellings -> our codes
ABBR_FIX = {"AZ": "ARI", "ATH": "OAK", "CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF",
            "TBR": "TB", "WAS": "WSH", "WSN": "WSH"}

FULL_NAME = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "OAK", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "Seattle Mariners": "SEA", "San Francisco Giants": "SF", "St. Louis Cardinals": "STL",
    "St Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}


def std_abbr(x: str) -> str:
    x = str(x).strip()
    return FULL_NAME.get(x) or ABBR_FIX.get(x.upper(), x.upper())


def roof_closed(team: str, temp_f: float | None, precip_prob: float | None) -> bool:
    roof = PARKS[team][3]
    if roof == "dome":
        return True
    if roof == "open":
        return False
    if precip_prob is not None and precip_prob >= 40:
        return True
    if temp_f is None:
        return True
    # hot-weather parks close for heat, cold-weather parks for cold
    return temp_f > 85 or temp_f < 62
