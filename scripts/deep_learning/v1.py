import os
import numpy as np
from PIL import Image, ImageDraw
import networkx as nx
import osmnx as ox
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from sklearn.metrics.pairwise import euclidean_distances

# ----------------------------
# 1️⃣ Génération des graphes NetworkX pour une liste de villes
# ----------------------------
def generate_graphs_for_cities(city_list, network_type='drive'):
    """
    city_list: liste de noms de villes
    Retourne dict {ville: NetworkX.Graph}
    """
    city_graphs = {}
    for city_name in city_list:
        print(f"Récupération du graphe pour {city_name}...")
        G = ox.graph_from_place(city_name, network_type=network_type, simplify=True)  # déjà simplifié
        city_graphs[city_name] = G
    return city_graphs

# ----------------------------
# 2️⃣ Génération des masques routes/intersections
# ----------------------------
def graph_to_mask(G, img_size=(512, 512)):
    width, height = img_size
    mask_routes = Image.new("L", (width, height), 0)
    mask_inter  = Image.new("L", (width, height), 0)

    xs = [data["x"] for _, data in G.nodes(data=True)]
    ys = [data["y"] for _, data in G.nodes(data=True)]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # ----------------------
    # Fonction locale pour projeter lon/lat -> pixels
    # ----------------------
    def project(lon, lat):
        x = int(round((lon - min_x) / (max_x - min_x) * (width - 1)))
        y = int(round((lat - min_y) / (max_y - min_y) * (height - 1)))
        x = max(0, min(x, width-1))
        y = max(0, min(y, height-1))
        return x, height - y  # inversion verticale pour Y

    # Dessiner les routes
    draw_routes = ImageDraw.Draw(mask_routes)
    for u, v in G.edges():
        x1, y1 = project(G.nodes[u]["x"], G.nodes[u]["y"])
        x2, y2 = project(G.nodes[v]["x"], G.nodes[v]["y"])
        draw_routes.line([x1, y1, x2, y2], fill=255, width=2)

    # Dessiner les intersections
    draw_inter = ImageDraw.Draw(mask_inter)
    for n, data in G.nodes(data=True):
        if G.degree[n] > 2:
            x, y = project(data["x"], data["y"])
            r = 3
            draw_inter.ellipse([x-r, y-r, x+r, y+r], fill=255)

    # Forcer taille exacte
    mask_routes = mask_routes.crop((0, 0, width, height))
    mask_inter  = mask_inter.crop((0, 0, width, height))

    return mask_routes, mask_inter



# ----------------------------
# 3️⃣ Dataset PyTorch
# ----------------------------
class RoadDataset(Dataset):
    def __init__(self, images, graphs, transform=None):
        self.images = images
        self.graphs = graphs
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        G   = self.graphs[idx]

        # Masques PIL
        mask_routes, mask_inter = graph_to_mask(G, img_size=img.size)

        # Image tensor (C,H,W)
        img_array = np.array(img, dtype=np.uint8)
        if img_array.ndim == 2:
            img_array = np.stack([img_array]*3, axis=-1)
        img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2,0,1) / 255.0

        # Masques tensor
        mask_routes = torch.tensor(np.array(mask_routes, dtype=np.uint8), dtype=torch.float32).unsqueeze(0) / 255.0
        mask_inter  = torch.tensor(np.array(mask_inter,  dtype=np.uint8), dtype=torch.float32).unsqueeze(0) / 255.0

        # ⚡ Forcer la taille des masques à celle de l'image
        mask_routes = torch.nn.functional.interpolate(mask_routes.unsqueeze(0),
                                                    size=(img_tensor.shape[1], img_tensor.shape[2]),
                                                    mode='nearest').squeeze(0)
        mask_inter  = torch.nn.functional.interpolate(mask_inter.unsqueeze(0),
                                                    size=(img_tensor.shape[1], img_tensor.shape[2]),
                                                    mode='nearest').squeeze(0)

        return img_tensor, mask_routes, mask_inter




# ----------------------------
# 4️⃣ Simple UNet pour route/intersections
# ----------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, out_channels, 1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.up(e2)
        out = self.dec1(d1)
        return torch.sigmoid(out)

# ----------------------------
# 5️⃣ Entraînement des CNN
# ----------------------------
def train_model(model, dataset, save_path, epochs=5, lr=1e-3):
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()
    for epoch in range(epochs):
        for img, mask_routes, mask_inter in loader:
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, mask_routes)  # ou mask_inter
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé dans {save_path}")

# ----------------------------
# 6️⃣ Mise en correspondance intersections pour géolocalisation
# ----------------------------
def match_intersections(pred_mask, city_graphs):
    pred_coords = np.argwhere(pred_mask > 0.5)
    best_city = None
    best_score = float('inf')
    for city, G in city_graphs.items():
        graph_coords = np.array(list(G.nodes()))
        if len(graph_coords) == 0: continue
        x_min, y_min = graph_coords.min(axis=0)
        x_max, y_max = graph_coords.max(axis=0)
        graph_coords_norm = np.array([
            (int((x-x_min)/(x_max-x_min)*(pred_mask.shape[1]-1)),
             int((y-y_min)/(y_max-y_min)*(pred_mask.shape[0]-1)))
            for x, y in graph_coords
        ])
        dists = euclidean_distances(pred_coords, graph_coords_norm)
        score = np.mean(np.min(dists, axis=1))
        if score < best_score:
            best_score = score
            best_city = city
    return best_city

# ----------------------------
# 7️⃣ Pipeline principal
# ----------------------------
def main():
    # ---- Liste de villes à traiter ----
    cities = ["Cherbourg-en-Cotentin, France", "Lyon, France"]
    city_graphs = generate_graphs_for_cities(cities)

    # ---- Images d'entraînement correspondantes ----
    # Liste des chemins vers les images satellites/plans
    train_image_paths = ["/home/cmoinier/Documents/r&d_ORT/images_reference/cherbourg.png", "/home/cmoinier/Documents/r&d_ORT/images_reference/lyon.png"]

    # Charger les images avec PIL
    images = [Image.open(path).convert("RGB") for path in train_image_paths]
    dataset = RoadDataset(images, [city_graphs[c] for c in cities], transform=T.ToTensor())

    # ---- Entraîner CNN routes ----
    model_routes = SimpleUNet()
    train_model(model_routes, dataset, save_path="model_routes.pth", epochs=2)

    # ---- Entraîner CNN intersections ----
    # Pour simplification, utiliser le même dataset et modèle
    model_inter = SimpleUNet()
    train_model(model_inter, dataset, save_path="model_intersections.pth", epochs=2)

    # ---- Exemple d'inférence sur une image inconnue ----
    test_img = Image.new("RGB", (512, 512), 200)  # remplacer par vraie image
    transform = T.ToTensor()
    test_tensor = transform(test_img).unsqueeze(0)
    model_routes.load_state_dict(torch.load("model_routes.pth"))
    model_routes.eval()
    with torch.no_grad():
        pred_routes = model_routes(test_tensor)[0,0].numpy()
    model_inter.load_state_dict(torch.load("model_intersections.pth"))
    model_inter.eval()
    with torch.no_grad():
        pred_inter = model_inter(test_tensor)[0,0].numpy()

    # ---- Géolocalisation automatique ----
    detected_city = match_intersections(pred_inter, city_graphs)
    print("Ville détectée :", detected_city)

if __name__ == "__main__":
    main()
