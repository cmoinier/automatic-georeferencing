import fiona
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import math
from shapely.geometry import shape

# Charger le GeoJSON
with fiona.open("france_lines.geojson") as src:
    roads = [feat["geometry"] for feat in src]
    bounds = src.bounds  # xmin, ymin, xmax, ymax

# Paramètres
pixel_size = 500  # m/pixel
tile_size_m = 100000  # 100 km
nx_tiles = math.ceil((bounds[2] - bounds[0]) / tile_size_m)
ny_tiles = math.ceil((bounds[3] - bounds[1]) / tile_size_m)
buffered_roads = [shape(feat['geometry']).buffer(50) for feat in roads]

for i in range(nx_tiles):
    for j in range(ny_tiles):
        xmin = bounds[0] + i*tile_size_m
        ymin = bounds[1] + j*tile_size_m
        xmax = xmin + tile_size_m
        ymax = ymin + tile_size_m
        width = int(tile_size_m / pixel_size)
        height = int(tile_size_m / pixel_size)
        transform = from_origin(xmin, ymax, pixel_size, pixel_size)

        tile_raster = rasterize(
            [(geom, 1) for geom in roads],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype='uint8'
        )

        # Sauvegarder la tuile
        with rasterio.open(f"/tiles/tile_{i}_{j}.tif", "w",
                           driver="GTiff", height=height, width=width,
                           count=1, dtype=tile_raster.dtype,
                           crs=src.crs, transform=transform) as dst:
            dst.write(tile_raster, 1)
