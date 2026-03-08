"""
map/visualizer.py
Geocodes location entities and builds an interactive Folium crime hotspot map.
"""

import folium
from folium.plugins import HeatMap, MarkerCluster
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time


# Cache geocoding results to avoid repeated API calls
_geocache = {}


def geocode_location(location: str) -> tuple:
    """
    Convert a location name to (lat, lon) coordinates.
    Returns None if geocoding fails.
    """
    if location in _geocache:
        return _geocache[location]

    try:
        geolocator = Nominatim(user_agent="crime_news_analyzer_bara_luch")
        result = geolocator.geocode(location, timeout=5)
        time.sleep(0.5)  # Respect rate limits

        if result:
            coords = (result.latitude, result.longitude)
            _geocache[location] = coords
            return coords
    except (GeocoderTimedOut, GeocoderServiceError):
        pass

    _geocache[location] = None
    return None


def build_map(locations: list, analysis: dict) -> str:
    """
    Build an interactive Folium map with:
    - Markers for each geocoded location
    - Heatmap overlay for crime density
    - Color-coded by crime type
    - Popups with entity info

    Returns HTML string of the map, or None if no locations found.
    """
    if not locations:
        return None

    # Geocode all locations
    geocoded = []
    for loc in set(locations):  # Deduplicate
        coords = geocode_location(loc)
        if coords:
            geocoded.append({"name": loc, "lat": coords[0], "lon": coords[1]})

    if not geocoded:
        return None

    # Center map on mean of all coordinates
    center_lat = sum(g["lat"] for g in geocoded) / len(geocoded)
    center_lon = sum(g["lon"] for g in geocoded) / len(geocoded)

    # Determine zoom level based on spread
    lat_spread = max(g["lat"] for g in geocoded) - min(g["lat"] for g in geocoded)
    zoom = 10 if lat_spread < 1 else 6 if lat_spread < 10 else 3

    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB positron"
    )

    # Crime type color mapping
    crime_types = list(analysis.get("crime_types", {}).keys())
    crime_color_map = {
        "violence": "red",
        "terrorism": "darkred",
        "theft": "orange",
        "arrest": "blue",
        "drug": "purple",
    }
    marker_color = crime_color_map.get(crime_types[0], "cadetblue") if crime_types else "cadetblue"

    # Marker cluster for clean rendering
    cluster = MarkerCluster(name="Locations").add_to(m)

    for loc in geocoded:
        # Build popup content
        popup_html = f"""
        <div style="font-family: Arial; min-width: 150px;">
            <b style="font-size: 14px;">📍 {loc['name']}</b><br>
            <small style="color: #666;">Crime types: {', '.join(crime_types) if crime_types else 'N/A'}</small><br>
            <small>Lat: {loc['lat']:.4f}, Lon: {loc['lon']:.4f}</small>
        </div>
        """
        folium.Marker(
            location=[loc["lat"], loc["lon"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=loc["name"],
            icon=folium.Icon(color=marker_color, icon="exclamation-sign", prefix="glyphicon")
        ).add_to(cluster)

    # Add heatmap if enough points
    if len(geocoded) >= 2:
        heat_data = [[g["lat"], g["lon"], 1.0] for g in geocoded]
        HeatMap(
            heat_data,
            name="Crime Density Heatmap",
            min_opacity=0.4,
            radius=30,
            blur=20,
            gradient={0.4: "blue", 0.65: "orange", 1: "red"}
        ).add_to(m)

    # Layer control
    folium.LayerControl().add_to(m)

    # Legend
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 12px 16px; border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-family: Arial; font-size: 13px;">
        <b>🔍 Crime News Analyzer</b><br>
        <span style="color: #666;">Locations: {len(geocoded)}</span><br>
        <span style="color: #666;">Crime types: {', '.join(crime_types) if crime_types else 'N/A'}</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m._repr_html_()
