import fiona
import rasterio
from rasterio.features import rasterize

# Charger le GeoJSON
with fiona.open("/home/cmoinier/Documents/r&d_ORT/france_lines.geojson", "r") as src:
    crs = src.crs
    bounds = src.bounds
    geometries = [feat["geometry"] for feat in src]

# Définir résolution et transform
pixel_size = 10
width = int((bounds[2] - bounds[0]) / pixel_size)
height = int((bounds[3] - bounds[1]) / pixel_size)
transform = rasterio.transform.from_origin(bounds[0], bounds[3], pixel_size, pixel_size)

# Rasterisation
raster = rasterize(
    [(geom, 1) for geom in geometries],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8"
)

# Sauvegarde en GeoTIFF
with rasterio.open(
    "/home/cmoinier/Documents/r&d_ORT/france_routes.tif", "w",
    driver="GTiff",
    height=height, width=width,
    count=1, dtype=raster.dtype,
    crs=crs, transform=transform
) as dst:
    dst.write(raster, 1)
