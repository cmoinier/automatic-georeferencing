######## TEST AVEC RAPIDFUZZ

import cv2
import pytesseract
from rapidfuzz import process
import re

# Charger la liste de communes
with open("/home/cmoinier/Documents/r&d_ORT/communes_clean.txt", "r", encoding="utf-8") as f:
    villes = [line.strip() for line in f if line.strip()]

# 1. Prétraitement image
img = cv2.imread("/home/cmoinier/Documents/r&d_ORT/images_ORT/carte4.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Redimensionner
resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Binarisation
_, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Débruitage
denoised = cv2.medianBlur(thresh, 3)

# 2. OCR global
custom_config = "--oem 3 --psm 11"
text = pytesseract.image_to_string(denoised, lang="fra", config=custom_config)
print("Texte OCR brut :", text)

# 3. Filtrer : garder uniquement les mots avec ≥4 lettres consécutives
mots_valides = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{4,}", text)

print("\nMots retenus :", mots_valides)

# 4. Correction par fuzzy matching
for mot in mots_valides:
    correction, score, _ = RAPIDFUZZ.extractOne(mot, villes)
    if score > 80:  # seuil ajustable
        print(f"Mot OCR '{mot}' corrigé en '{correction}' (score {score})")