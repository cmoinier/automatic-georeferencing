import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os

# ------------------------------
# 1️⃣ Chemins
# ------------------------------
screenshot_path = "/images_ORT/carte1.png"
sentinel_path = "path_to_sentinel_image.tif"
out_path = "screenshot_aligne.tif"

# ------------------------------
# 2️⃣ Vérification du screenshot
# ------------------------------
if not os.path.exists(screenshot_path):
    raise FileNotFoundError(f"❌ Screenshot non trouvé : {screenshot_path}")

screenshot = cv2.imread(screenshot_path)
if screenshot is None:
    raise ValueError(f"❌ Impossible de lire l'image : {screenshot_path}")

# ------------------------------
# 3️⃣ Charger Sentinel-2 pour récupérer bbox et CRS
# ------------------------------
with rasterio.open(sentinel_path) as src:
    width, height = src.width, src.height
    bounds_s2 = src.bounds
    crs = src.crs

print(f"✅ Sentinel-2 chargé : width={width}, height={height}")

# ------------------------------
# 4️⃣ Redimensionner le screenshot à la bbox Sentinel-2
# ------------------------------
screenshot_resized = cv2.resize(screenshot, (width, height), interpolation=cv2.INTER_LINEAR)
print(f"✅ Screenshot redimensionné à la bbox Sentinel-2")

# ------------------------------
# 5️⃣ (Optionnel) appliquer une transformation affine si tu as des points de contrôle
# Ici, on ne fait rien pour garder l'image simplement redimensionnée
aligned = screenshot_resized

# ------------------------------
# 6️⃣ Sauvegarder GeoTIFF
# ------------------------------
transform_final = from_bounds(bounds_s2.left, bounds_s2.bottom, bounds_s2.right, bounds_s2.top, width, height)

with rasterio.open(
    out_path, 'w',
    driver='GTiff',
    height=height,
    width=width,
    count=3,
    dtype=aligned.dtype,
    crs=crs,
    transform=transform_final
) as dst:
    for i in range(3):
        dst.write(aligned[:,:,i], i+1)

print(f"✅ Screenshot aligné et géoréférencé sauvegardé : {out_path}")
