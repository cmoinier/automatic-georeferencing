import ee
import geemap
import os
import osmnx as ox

# ------------------------------
# 1️⃣ Authentification (si nécessaire)
# ------------------------------
try:
    ee.Initialize(project='computervision-470410')
except ee.EEException:
    ee.Authenticate()
    ee.Initialize(project='computervision-470410')
print("✅ GEE initialisé !")

# --- 2. Récupérer la bounding box de la commune ---
commune = "Beaumont-Hague, France"  # nom de la commune
gdf = ox.geocode_to_gdf(commune)
bbox = gdf.total_bounds
print("Bounding box:", bbox)
geometry = ee.Geometry.Polygon([
    [[bbox[0], bbox[1]],
     [bbox[2], bbox[1]],
     [bbox[2], bbox[3]],
     [bbox[0], bbox[3]],
     [bbox[0], bbox[1]]]
])

# ------------------------------
# 3️⃣ Récupérer la dernière image Sentinel-2 L1C
# ------------------------------
collection = (
    ee.ImageCollection("COPERNICUS/S2")
    .filterBounds(geometry)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .sort('system:time_start', False)  # plus récente en premier
)

image = collection.first()
if image is None:
    raise Exception("❌ Aucun produit Sentinel-2 trouvé pour cette zone.")

print("✅ Image trouvée :", image.get('PRODUCT_ID').getInfo())

# ------------------------------
# 4️⃣ Sélection des bandes RGB
# ------------------------------
rgb = image.select(['B4', 'B3', 'B2'])

# ------------------------------
# 5️⃣ Export direct sur disque local
# ------------------------------
out_dir = "./gee_download"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"sentinel2_rgb_{commune}.tif")

# geemap fournit ee_export_image pour exporter directement
geemap.ee_export_image(
    rgb,
    filename=out_path,
    scale=10,       # résolution Sentinel-2
    crs='EPSG:4326',
    region=geometry,
    file_per_band=False
)

print(f"✅ GeoTIFF téléchargé directement sur disque : {out_path}")
