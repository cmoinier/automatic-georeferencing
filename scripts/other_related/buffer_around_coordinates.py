from shapely.geometry import Point
import osmnx as ox

# Exemple : coordonnées depuis Nominatim
place_name = "Beaumont-Hague, France"
G = ox.graph_from_place(place_name, network_type='drive')

# Sauvegarder en GeoJSON
gdf_edges = ox.graph_to_gdfs(G, nodes=False)
gdf_edges.to_file("/home/cmoinier/Documents/r&d_ORT/routes_beaumont.geojson", driver="GeoJSON")
