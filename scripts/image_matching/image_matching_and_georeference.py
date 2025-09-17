import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
# Charger les deux images en niveaux de gris
img1 = cv2.imread("/map-image.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("/georeferenced-image.tif", cv2.IMREAD_GRAYSCALE)

# ------------------------------
# 1. Générer les descripteurs
# ------------------------------
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# ------------------------------
# 2. Matching des features
# ------------------------------
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Ratio test de Lowe
good = []
pts1, pts2 = [], []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.float32(pts1).reshape(-1, 1, 2)
pts2 = np.float32(pts2).reshape(-1, 1, 2)

print(f"Nombre de correspondances retenues : {len(good)}")

# ------------------------------
# 3. Calcul de l'homographie
# ------------------------------
if len(good) > 4:
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    print("Matrice d'homographie :\n", H)

    # Appliquer la transformation
    h, w = img1.shape
    warped = cv2.warpPerspective(img1, H, (w, h))

    # ------------------------------
    # 4. Export en GeoTIFF
    # ------------------------------
    with rasterio.open("/georeferenced-image.tif") as ref:
        profile = ref.profile.copy()  # on récupère métadonnées (CRS, transform, etc.)
        
        # Adapter le profil à la nouvelle image
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            height=warped.shape[0],
            width=warped.shape[1]
        )
        
        # Écrire le GeoTIFF aligné
        with rasterio.open("/aligned-map-image.tif", "w", **profile) as dst:
            dst.write(warped.astype(rasterio.uint8), 1)

    print("✅ Fichier 'aligned-map-image.tif' créé et géoréférencé")
    
    # ------------------------------
    # 5. Visualisation des correspondances
    # ------------------------------
    result = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(15, 8))
    plt.imshow(result)
    plt.axis("off")
    plt.show()

else:
    print("⚠️ Pas assez de correspondances fiables pour calculer une homographie")
