###### TEST AVEC EASYOCR

import easyocr
from difflib import get_close_matches
from geopy.geocoders import Nominatim
import unicodedata
import time

# Nettoyage pour OCR et comparaison
def normalize_text(text):
    text = text.lower()
    text = text.replace(" ", "").replace("-", "")
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')  # enlever accents
    return text

# Charger la liste des communes
with open("/other_resources/communes_clean.txt", "r", encoding="utf-8") as f:
    communes = [line.strip() for line in f.readlines()]

# Dictionnaire : clé normalisée -> valeur originale
communes_normalized = {normalize_text(c): c for c in communes}

# Fonction pour matcher
def match_commune(text):
    norm_text = normalize_text(text)
    # correspondance exacte
    if norm_text in communes_normalized:
        return communes_normalized[norm_text]
    # sinon correspondance proche
    matches = get_close_matches(norm_text, communes_normalized.keys(), n=1, cutoff=0.6)
    return communes_normalized[matches[0]] if matches else None


# OCR
reader = easyocr.Reader(['fr'])
results = reader.readtext("/images_ORT/carte4.jpg")

# Initialiser le géocodeur
geolocator = Nominatim(user_agent="myApp")

for (_, text, _) in results:
    matched_commune = match_commune(text)
    if matched_commune:
        print(f"OCR: {text} -> Commune trouvée: {matched_commune}")
        try:
            location = geolocator.geocode(f"{matched_commune}, France", timeout=10)
            if location:
                print(f"Latitude: {location.latitude}, Longitude: {location.longitude}")
            else:
                print("Géolocalisation non trouvée")
        except Exception as e:
            print(f"Erreur de géocodage pour {matched_commune} : {e}")
        time.sleep(1)  # respecter la limite de Nominatim
    else:
        print(f"OCR: {text} -> Aucune commune correspondante")
