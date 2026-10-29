"""Game-time temperature forecasts: Open-Meteo first, US National Weather Service as fallback (both free, no key)."""

from __future__ import annotations

from datetime import datetime

import requests

from .parks import PARKS, roof_closed

URL = "https://api.open-meteo.com/v1/forecast"


def game_weather(team: str, start_utc: str) -> dict:
    """Forecast for the home park at first pitch. Falls back to 'unknown' (dome-like 72F) on failure."""
    _, lat, lon, _ = PARKS[team]
    t = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
    try:
        r = requests.get(URL, params={"latitude": lat, "longitude": lon, "hourly": "temperature_2m,precipitation_probability",
                                      "temperature_unit": "fahrenheit", "timezone": "UTC",
                                      "start_date": t.strftime("%Y-%m-%d"), "end_date": t.strftime("%Y-%m-%d")},
                         timeout=20)
        r.raise_for_status()
        h = r.json()["hourly"]
        idx = h["time"].index(t.strftime("%Y-%m-%dT%H:00"))
        temp, pp = float(h["temperature_2m"][idx]), h.get("precipitation_probability", [None] * 24)[idx]
    except Exception:  # noqa: BLE001
        temp, pp = _nws(lat, lon, t)
    closed = roof_closed(team, temp, pp)
    local_hour = (t.hour + round(lon / 15) + 1) % 24   # rough local time (daylight saving)
    night = float(local_hour >= 17)
    return {"temp": 72.0 if closed or temp is None else temp, "wind_out": 0.0, "wind_in": 0.0,
            "dome": float(closed), "night": night, "forecast_temp": temp, "precip_prob": pp}


def _nws(lat, lon, t):
    """api.weather.gov hourly forecast (US parks only). Returns (temp_f, precip_prob) or (None, None)."""
    try:
        hdr = {"User-Agent": "moneyball-runner", "Accept": "application/geo+json"}
        pt = requests.get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}", headers=hdr, timeout=20).json()
        fc = requests.get(pt["properties"]["forecastHourly"], headers=hdr, timeout=20).json()
        for per in fc["properties"]["periods"]:
            start = datetime.fromisoformat(per["startTime"])
            if start <= t < datetime.fromisoformat(per["endTime"]):
                temp = float(per["temperature"])
                if per.get("temperatureUnit") == "C":
                    temp = temp * 9 / 5 + 32
                return temp, (per.get("probabilityOfPrecipitation") or {}).get("value")
    except Exception:  # noqa: BLE001
        pass
    return None, None
