import cv2
import numpy as np
import networkx as nx
from shapely.geometry import LineString, mapping
import geojson

# --- 1. Charger l'image ---
img = cv2.imread("/images_ORT/carte4.jpg")
if img is None:
    raise FileNotFoundError("Image non trouvée ! Vérifie le chemin.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 2. Binarisation ---
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("/binary_hague.png", binary)

# --- 3. Skeletonisation ---
skeleton = cv2.ximgproc.thinning(binary)
cv2.imwrite("/skeleton_hague.png", skeleton)

# --- 4. Identifier les nœuds (extrémités et intersections) ---
def get_neighbors(img, x, y):
    return img[y-1:y+2, x-1:x+2].sum() // 255 - 1

nodes = {}
node_id = 0
for y in range(1, skeleton.shape[0]-1):
    for x in range(1, skeleton.shape[1]-1):
        if skeleton[y, x] == 255:
            n = get_neighbors(skeleton, x, y)
            if n == 1 or n >= 3:  # extrémité ou intersection
                nodes[(x, y)] = node_id
                node_id += 1

# --- 5. Suivre les arêtes ---
visited = np.zeros_like(skeleton, dtype=bool)
edges = []

def trace_edge(x, y, prev_pt):
    path = [(x, y)]
    cx, cy = x, y
    while True:
        visited[cy, cx] = True
        neighbors = []
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                nx_, ny_ = cx+dx, cy+dy
                if (0 <= nx_ < skeleton.shape[1] and 0 <= ny_ < skeleton.shape[0]
                    and skeleton[ny_, nx_] == 255 and not visited[ny_, nx_]):
                    neighbors.append((nx_, ny_))
        if not neighbors:
            break
        # prendre le premier voisin (simple pour l'exemple)
        cx, cy = neighbors[0]
        path.append((cx, cy))
        if (cx, cy) in nodes and (cx, cy) != prev_pt:
            break
    return path

G = nx.Graph()

# Ajouter les nœuds au graphe
for pt, nid in nodes.items():
    G.add_node(nid, pos=pt)

# Tracer les arêtes
for pt, nid in nodes.items():
    if not visited[pt[1], pt[0]]:
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                nx_, ny_ = pt[0]+dx, pt[1]+dy
                if (0 <= nx_ < skeleton.shape[1] and 0 <= ny_ < skeleton.shape[0]
                    and skeleton[ny_, nx_] == 255 and (nx_, ny_) not in nodes):
                    path = trace_edge(nx_, ny_, pt)
                    if path[-1] in nodes:
                        G.add_edge(nid, nodes[path[-1]], geometry=LineString(path))

# --- 6. Export GeoJSON ---
features = []
for u, v, data in G.edges(data=True):
    features.append(geojson.Feature(geometry=mapping(data["geometry"]), properties={}))

with open("/graph_hague.geojson", "w") as f:
    geojson.dump(geojson.FeatureCollection(features), f)

print("Graphe vectoriel exporté en graph.geojson ✅")
