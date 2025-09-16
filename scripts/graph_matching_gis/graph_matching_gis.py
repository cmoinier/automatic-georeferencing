import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from sklearn.metrics.pairwise import euclidean_distances

# ----------------------------
# 1️⃣ Fonction pour générer masque à partir d'un graphe
# ----------------------------
def graph_to_mask(G, img_size=(512, 512)):
    """
    G: graphe NetworkX avec noeuds en (x, y)
    img_size: taille de l'image (pixels)
    Retourne un masque binaire avec routes et intersections
    """
    x_vals = [x for x, y in G.nodes()]
    y_vals = [y for x, y in G.nodes()]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    def normalize(x, y):
        x_norm = int((x - x_min) / (x_max - x_min) * (img_size[0] - 1))
        y_norm = int((y - y_min) / (y_max - y_min) * (img_size[1] - 1))
        return x_norm, y_norm

    mask_routes = Image.new("L", img_size, 0)
    mask_intersections = Image.new("L", img_size, 0)
    draw_routes = ImageDraw.Draw(mask_routes)
    draw_inter = ImageDraw.Draw(mask_intersections)

    # Dessiner routes
    for u, v in G.edges():
        draw_routes.line([normalize(*u), normalize(*v)], fill=255, width=2)

    # Dessiner intersections
    for node in G.nodes():
        x, y = normalize(*node)
        draw_inter.ellipse([x-2, y-2, x+2, y+2], fill=255)

    return np.array(mask_routes), np.array(mask_intersections)

# ----------------------------
# 2️⃣ Dataset PyTorch
# ----------------------------
class RoadDataset(Dataset):
    def __init__(self, images, graphs, transform=None):
        """
        images: liste de PIL.Image ou chemins
        graphs: liste de NetworkX.Graph correspondant aux images
        """
        self.images = images
        self.graphs = graphs
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask_routes, mask_inter = graph_to_mask(self.graphs[idx], img_size=img.size)
        if self.transform:
            img = self.transform(img)
            mask_routes = torch.tensor(mask_routes, dtype=torch.float32).unsqueeze(0) / 255.0
            mask_inter = torch.tensor(mask_inter, dtype=torch.float32).unsqueeze(0) / 255.0
        return img, mask_routes, mask_inter

# ----------------------------
# 3️⃣ CNN simple type UNet pour route ou intersections
# ----------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(16, 16, 3, padding=1),
                                  nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, padding=1),
                                  nn.ReLU())
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(16, out_channels, 1))
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.up(e2)
        out = self.dec1(d1)
        return torch.sigmoid(out)

# ----------------------------
# 4️⃣ Exemple d'entraînement
# ----------------------------
def train_model(model, dataset, epochs=5, lr=1e-3):
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

# ----------------------------
# 5️⃣ Exemple de mise en correspondance des intersections pour géolocalisation
# ----------------------------
def match_intersections(pred_mask, city_graphs):
    """
    pred_mask: masque binaire intersections prédit par le CNN
    city_graphs: dict {nom_ville: NetworkX.Graph}
    Retourne le nom de la ville la plus probable
    """
    # Récupérer les coordonnées des intersections détectées
    pred_coords = np.argwhere(pred_mask > 0.5)  # pixels (y, x)
    
    best_city = None
    best_score = float('inf')
    for city, G in city_graphs.items():
        graph_coords = np.array(list(G.nodes()))
        # Normaliser graph_coords dans la taille du masque
        x_min, y_min = graph_coords.min(axis=0)
        x_max, y_max = graph_coords.max(axis=0)
        graph_coords_norm = np.array([(int((x-x_min)/(x_max-x_min)*(pred_mask.shape[1]-1)),
                                       int((y-y_min)/(y_max-y_min)*(pred_mask.shape[0]-1))) 
                                       for x, y in graph_coords])
        if len(graph_coords_norm) == 0:
            continue
        # Distance minimale moyenne (simplifiée)
        dists = euclidean_distances(pred_coords, graph_coords_norm)
        score = np.mean(np.min(dists, axis=1))
        if score < best_score:
            best_score = score
            best_city = city
    return best_city

# ----------------------------
# 6️⃣ Exemple d'utilisation
# ----------------------------
if __name__ == "__main__":
    # Images et graphes factices pour l'exemple
    img = Image.new("RGB", (512, 512), 0)
    G = nx.grid_2d_graph(10, 10)  # graphe fictif pour test
    dataset = RoadDataset([img], [G], transform=T.ToTensor())
    
    model_routes = SimpleUNet()
    train_model(model_routes, dataset, epochs=1)  # juste pour test

    # Exemple de mise en correspondance
    pred_mask = np.random.rand(512, 512)  # ici remplacer par la sortie réelle du CNN
    city_graphs = {"ville_test": G}
    city_name = match_intersections(pred_mask, city_graphs)
    print("Ville détectée:", city_name)
