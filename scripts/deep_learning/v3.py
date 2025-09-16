import os
import numpy as np
from PIL import Image, ImageDraw
import networkx as nx
import osmnx as ox
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ Génération des graphes NetworkX pour une liste de villes
# ----------------------------
def generate_graphs_for_cities(city_list, network_type='drive'):
    city_graphs = {}
    for city_name in city_list:
        if city_name not in city_graphs:
            print(f"Récupération du graphe pour {city_name}...")
            G = ox.graph_from_place(city_name, network_type=network_type)
            G = nx.Graph(G)  # graphe non dirigé
            if not G.graph.get("simplified", False):
                G = ox.simplify_graph(G)
            city_graphs[city_name] = G
    return city_graphs

# ----------------------------
# 2️⃣ Génération des masques routes/intersections
# ----------------------------
def graph_to_mask(G, img_size=(512,512)):
    width, height = img_size
    mask_routes = Image.new("L", (width, height), 0)
    mask_inter  = Image.new("L", (width, height), 0)

    xs = [data["x"] for _, data in G.nodes(data=True)]
    ys = [data["y"] for _, data in G.nodes(data=True)]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def project(lon, lat):
        x = int(round((lon - min_x) / (max_x - min_x) * (width - 1)))
        y = int(round((lat - min_y) / (max_y - min_y) * (height - 1)))
        x = max(0, min(x, width-1))
        y = max(0, min(y, height-1))
        return x, height - y  # inversion verticale

    draw_routes = ImageDraw.Draw(mask_routes)
    for u,v in G.edges():
        x1, y1 = project(G.nodes[u]["x"], G.nodes[u]["y"])
        x2, y2 = project(G.nodes[v]["x"], G.nodes[v]["y"])
        draw_routes.line([x1, y1, x2, y2], fill=255, width=2)

    draw_inter = ImageDraw.Draw(mask_inter)
    for n, data in G.nodes(data=True):
        if G.degree[n] > 2:
            x, y = project(data["x"], data["y"])
            r = 3
            draw_inter.ellipse([x-r, y-r, x+r, y+r], fill=255)

    return mask_routes, mask_inter

# ----------------------------
# 3️⃣ Dataset PyTorch
# ----------------------------
class RoadDataset(Dataset):
    def __init__(self, images, graphs):
        self.images = images
        self.graphs = graphs
        self.target_size = (512, 512)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].resize(self.target_size, Image.BILINEAR)
        G   = self.graphs[idx]

        mask_routes, mask_inter = graph_to_mask(G, img_size=self.target_size)

        img_tensor = torch.tensor(np.array(img, dtype=np.uint8), dtype=torch.float32).permute(2,0,1)/255.0
        mask_routes = torch.tensor(np.array(mask_routes, dtype=np.uint8), dtype=torch.float32).unsqueeze(0)/255.0
        mask_inter  = torch.tensor(np.array(mask_inter, dtype=np.uint8), dtype=torch.float32).unsqueeze(0)/255.0

        return img_tensor, mask_routes, mask_inter

# ----------------------------
# 4️⃣ Simple UNet pour route/intersections
# ----------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1), nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.ReLU()
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,out_channels,1)
        )

    def forward(self,x):
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
            if output.shape != mask_routes.shape:
                mask_routes = F.interpolate(mask_routes, size=output.shape[2:], mode='nearest')
            loss = criterion(output, mask_routes)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé dans {save_path}")

# ----------------------------
# 6️⃣ Géolocalisation via intersections
# ----------------------------
def match_intersections(pred_mask, city_graphs):
    pred_coords = np.argwhere(pred_mask > 0.5)
    best_city = None
    best_score = float('inf')
    for city, G in city_graphs.items():
        graph_coords = np.array([[data["x"], data["y"]] for _, data in G.nodes(data=True)])
        if len(graph_coords)==0: continue
        x_min, y_min = graph_coords.min(axis=0)
        x_max, y_max = graph_coords.max(axis=0)
        graph_coords_norm = np.array([
            (int(round((x-x_min)/(x_max-x_min)*(pred_mask.shape[1]-1))),
             int(round((y-y_min)/(y_max-y_min)*(pred_mask.shape[0]-1))))
            for x,y in graph_coords
        ])
        print(f"Pred coords: {len(pred_coords)}, Graph coords: {len(graph_coords)}")
        if len(pred_coords)==0: 
            continue
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
    # 1️⃣ Liste de villes et génération des graphes
    city_names = ["Cherbourg-en-Cotentin, France", "Lyon, France", "Beaumont-Hague, France", "Lille, France"]
    city_graphs = generate_graphs_for_cities(city_names)

    # 2️⃣ Images d'entraînement (10 par ville)
    image_paths = [
        # 10 Cherbourg
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/cherb1.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/cherb2.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/cherb3.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/cherb4.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/cherb5.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/cherb6.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/cherb7.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/cherb8.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/cherb9.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/cherb10.png",
        # 10 Lyon
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon1.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon2.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon3.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon4.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon5.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon6.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon7.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon8.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon9.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon10.png",
    ]
    images = [Image.open(p).convert("RGB") for p in image_paths]

    # 3️⃣ Associer chaque image à son graphe
    graphs_for_dataset = [city_graphs["Cherbourg-en-Cotentin, France"]] * 10 + \
                         [city_graphs["Lyon, France"]] * 10

    # 4️⃣ Dataset PyTorch
    dataset = RoadDataset(images, graphs_for_dataset)

    # 5️⃣ Entraînement
    model_routes = SimpleUNet()
    train_model(model_routes, dataset, save_path="model_routes.pth", epochs=2)
    model_inter = SimpleUNet()
    train_model(model_inter, dataset, save_path="model_intersections.pth", epochs=2)

    # 6️⃣ Test sur une image
    test_img_path = "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon1.png"
    test_img = Image.open(test_img_path).convert("RGB").resize((512,512))
    test_tensor = torch.tensor(np.array(test_img), dtype=torch.float32).permute(2,0,1).unsqueeze(0)/255.0

    # Inférence routes
    model_routes.load_state_dict(torch.load("model_routes.pth"))
    model_routes.eval()
    with torch.no_grad():
        pred_routes = model_routes(test_tensor)[0,0].numpy()

    # Inférence intersections
    model_inter.load_state_dict(torch.load("model_intersections.pth"))
    model_inter.eval()
    with torch.no_grad():
        pred_inter = model_inter(test_tensor)[0,0].numpy()

    # Visualisation
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(pred_routes, cmap="gray")
    plt.title("Masque routes prédit")
    plt.subplot(1,2,2)
    plt.imshow(pred_inter, cmap="gray")
    plt.title("Masque intersections prédit")
    plt.show()

    # Géolocalisation
    detected_city = match_intersections(pred_inter, city_graphs)
    print("Ville détectée :", detected_city)

if __name__=="__main__":
    main()
