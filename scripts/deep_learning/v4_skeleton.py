import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances

# ----------------------------
# 1️⃣ Fonction pour créer un mask skeletonisé à partir d'une image
# ----------------------------
def generate_skeleton_mask(img_path, target_size=(512,512)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image non trouvée: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binarisation
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # Skeletonisation
    skeleton = cv2.ximgproc.thinning(binary)
    # Redimensionner et convertir en PIL pour compatibilité PyTorch
    skeleton_pil = Image.fromarray(skeleton).resize(target_size)
    tensor = torch.tensor(np.array(skeleton_pil), dtype=torch.float32).unsqueeze(0)/255.0
    return tensor

# ----------------------------
# 2️⃣ Dataset PyTorch à partir d'images -> skeleton
# ----------------------------
class SkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, target_size=(512,512)):
        self.image_paths = image_paths
        self.target_size = target_size  # (width, height)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image non trouvée : {path}")

        # Redimensionne l'image
        img = cv2.resize(img, self.target_size)

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarisation (blanc = routes, noir = reste)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Skeletonisation
        skeleton = cv2.ximgproc.thinning(binary)

        # Normalisation et conversion en tensor float
        mask_tensor = torch.tensor(skeleton, dtype=torch.float32).unsqueeze(0)/255.0

        # Image tensor en RGB [C,H,W] si tu veux éventuellement la passer au CNN
        img_tensor = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=torch.float32).permute(2,0,1)/255.0

        return img_tensor, mask_tensor



# ----------------------------
# 3️⃣ UNet simple pour intersections
# ----------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
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
# 4️⃣ Entraînement
# ----------------------------
def train_model(model, dataset, save_path, epochs=5, lr=1e-3, use_img_input=False):
    """
    model        : ton UNet
    dataset      : SkeletonDataset
    save_path    : chemin pour sauvegarder le modèle
    epochs       : nb d'époques
    lr           : learning rate
    use_img_input: si True, le modèle prend l'image RGB en entrée, sinon le mask skeletonisé
    """
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()

    for epoch in range(epochs):
        for img_batch, mask_batch in loader:
            optimizer.zero_grad()

            # Choix de l'entrée
            input_tensor = img_batch if use_img_input else mask_batch

            output = model(input_tensor)

            # Redimensionner le masque si nécessaire
            if output.shape != mask_batch.shape:
                mask_batch = F.interpolate(mask_batch, size=output.shape[2:], mode='nearest')

            loss = criterion(output, mask_batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé dans {save_path}")



# ----------------------------
# 5️⃣ Géolocalisation intersections
# ----------------------------
def match_intersections(pred_mask, city_graphs):
    pred_coords = np.argwhere(pred_mask > 0.5)
    best_city = None
    best_score = float('inf')
    for city, G in city_graphs.items():
        graph_coords = np.array([[data["x"], data["y"]] for _, data in G.nodes(data=True)])
        if len(graph_coords)==0 or len(pred_coords)==0:
            continue
        x_min, y_min = graph_coords.min(axis=0)
        x_max, y_max = graph_coords.max(axis=0)
        graph_coords_norm = np.array([
            (int(round((x-x_min)/(x_max-x_min)*(pred_mask.shape[1]-1))),
             int(round((y-y_min)/(y_max-y_min)*(pred_mask.shape[0]-1))))
            for x,y in graph_coords
        ])
        dists = euclidean_distances(pred_coords, graph_coords_norm)
        score = np.mean(np.min(dists, axis=1))
        if score < best_score:
            best_score = score
            best_city = city
    return best_city

# ----------------------------
# 6️⃣ Pipeline principal
# ----------------------------
def main():
    # Villes d'entraînement et graphes pour la géolocalisation
    cities = ["Cherbourg-en-Cotentin, France", "Lyon, France"]
    city_graphs = {}
    for c in cities:
        G = nx.Graph()  # Remplacer par ox.graph_from_place(c) si tu veux OSM
        city_graphs[c] = G

    # Chemins vers les screenshots pour chaque ville (10 par ville)
    image_paths = [
        # Cherbourg
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
        # Lyon
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon1.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon2.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon3.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon4.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon5.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon6.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon7.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon8.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon9.png",
        "/home/cmoinier/Documents/r&d_ORT/lyon_cherb/lyon10.png"
    ]

    # Dataset
    dataset = SkeletonDataset(image_paths)

    # Modèle intersections
    model_inter = SimpleUNet(in_channels=1, out_channels=1)
    train_model(model_inter, dataset, save_path="model_intersections.pth", epochs=5)

    # Test sur une image inconnue
    test_mask_path = "/home/cmoinier/Documents/r&d_ORT/generated_training_images/cherb_sat.png"
    test_tensor = generate_skeleton_mask(test_mask_path).unsqueeze(0)  # [1,1,H,W]

    model_inter.eval()
    with torch.no_grad():
        pred_inter = model_inter(test_tensor)[0,0].numpy()

    detected_city = match_intersections(pred_inter, city_graphs)
    print("Ville détectée :", detected_city)


if __name__=="__main__":
    main()