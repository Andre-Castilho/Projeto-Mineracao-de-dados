# === PyTorch e Treinamento ===
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# === Visão Computacional ===
from torchvision import transforms, models
from torchvision.datasets import Omniglot
from PIL import Image

# === Utilidades ===
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import product


# === Clusterização e Avaliação ===
from sklearn.cluster import KMeans, BisectingKMeans 
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
)

# === Visualização ===
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

# Configuração do dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset_real = Omniglot(root="./data", background=True, transform=transform, download=True)
dataset_fictional = Omniglot(root="./data", background=False, transform=transform, download=True)

base_path = Path("./data/omniglot-py/")

if not (base_path / "images_background").exists() or not (base_path / "images_evaluation").exists():
    print(f"Error: Data directories not found in {base_path}")
    print("Omniglot download may have failed or is incomplete.")
    print("Try deleting the './data' folder and re-running the script.")
    exit()

real_alphabets = sorted([d.name for d in (base_path / "images_background").iterdir() if d.is_dir()])
fictional_alphabets = sorted([d.name for d in (base_path / "images_evaluation").iterdir() if d.is_dir()])

train_alphabets = real_alphabets

test_alphabets = fictional_alphabets

num_classes = len(train_alphabets) 

#=== Dataset Customizado para Omniglot ===
class OmniglotSplit(Dataset):
    def __init__(self, base_folder, target_alphabets, transform):
        self.transform = transform
        self.base_folder = Path(base_folder)
        self.target_alphabets = set(target_alphabets)
        
        self.alphabet_to_local_idx = {name: i for i, name in enumerate(sorted(list(self.target_alphabets)))}
        
        self.image_paths = []
        self.image_labels = []

        if not self.base_folder.exists():
            print(f"Error: Directory not found: {self.base_folder}")
            return

        for alphabet_dir in self.base_folder.iterdir():
            if alphabet_dir.is_dir() and alphabet_dir.name in self.target_alphabets:
                alphabet_label = self.alphabet_to_local_idx[alphabet_dir.name]
                for char_dir in alphabet_dir.iterdir():
                    for img_path in char_dir.glob('*.png'):
                        self.image_paths.append(img_path)
                        self.image_labels.append(alphabet_label)
                            
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.image_labels[idx]
        
        image = Image.open(img_path).convert('L') 
        if self.transform:
            image = self.transform(image)
            
        return image, label


test_folder = base_path / "images_evaluation"
test_dataset = OmniglotSplit(test_folder, test_alphabets, transform)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model_save_path = "resnet50.pth"
model = models.resnet50(weights=None)


print(f"\nLoading the model from: {model_save_path}")
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes) 
model.load_state_dict(torch.load(model_save_path, map_location=torch.device(device)))
model = model.to(device)
model.eval()

all_embeddings = []
all_labels = [] 

# Extrair embeddings do conjunto de teste
feature_extractor = nn.Sequential(*list(model.children())[:-1])

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Extracting Test Embeddings"):
        images = images.to(device)
        feats = feature_extractor(images)
        feats = torch.flatten(feats, 1) 
        all_embeddings.extend(feats.cpu().numpy())
        all_labels.extend(labels.cpu().numpy()) 

y_alphabet_test = [test_alphabets[i] for i in all_labels]
X_embeddings = np.array(all_embeddings)
all_labels = np.array(all_labels) 

X, y = X_embeddings, all_labels

#=== UMAP para Redução de Dimensionalidade ===
print("\nRunning UMAP on the test set embeddings")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_2d = reducer.fit_transform(X_embeddings)

#=== Plot UMAP com Rótulos Verdadeiros ===
print("Generating UMAP plot (True Labels)")

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=X_2d[:, 0], y=X_2d[:, 1],
    hue=y_alphabet_test, 
    palette='tab20', 
    s=15, 
    legend="full"
)
plt.title("UMAP Projection of Unseen Test Alphabet Embeddings (True Labels)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., markerscale=2, fontsize='small')
plt.tight_layout()
plt.savefig("umap_true_labels.png") 
plt.close() 
print("Saved umap_true_labels.png")

#=== Função de Avaliação de Clusterização ===
def evaluate_clustering(name, labels, X, y_true=None, X_2d=None):
    metrics = {}
    n_clusters = len(np.unique(labels))
    metrics["Algorithm"] = name
    metrics["Clusters"] = n_clusters

    
    if n_clusters > 1 and len(np.unique(labels)) > 1:
        try:
            metrics["Silhouette"] = silhouette_score(X, labels)
            metrics["CalinskiHarabasz"] = calinski_harabasz_score(X, labels)
            metrics["DaviesBouldin"] = davies_bouldin_score(X, labels)
        except Exception:
            metrics["Silhouette"] = np.nan
            metrics["CalinskiHarabasz"] = np.nan
            metrics["DaviesBouldin"] = np.nan
    else:
        metrics["Silhouette"] = np.nan
        metrics["CalinskiHarabasz"] = np.nan
        metrics["DaviesBouldin"] = np.nan

    
    if y_true is not None:
        metrics["ARI"] = adjusted_rand_score(y_true, labels)
        metrics["NMI"] = normalized_mutual_info_score(y_true, labels)
    else:
        metrics["ARI"] = np.nan
        metrics["NMI"] = np.nan

    print(f"\n{name} Results:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    
    if X_2d is not None:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            hue=labels.astype(str),
            palette="tab20",
            s=20,
            alpha=0.8,
            edgecolor=None,
            linewidth=0,
            legend="full"
        )
        plt.title(f"UMAP Projection ({name} Clusters)")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.axis("off")
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.,
            markerscale=2,
            fontsize='small'
        )
        plt.tight_layout()
        plt.savefig(f"umap_{name.lower().replace(' ', '_')}_clusters.png")
        plt.close()
        print(f"Saved umap_{name.lower().replace(' ', '_')}_clusters.png")

    return metrics

best_k = len(np.unique(y_alphabet_test))
X = X.astype(np.float64)  

#=== KMeans com Vários Valores de k ===

print("\n--- KMeans ---")
L = []
for k in range(2, 12):
    kmeans = KMeans(n_clusters=k, n_init=10, init='random', max_iter=300, random_state=42)
    kmeans.fit(X)
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    ari = adjusted_rand_score(y_alphabet_test, kmeans.labels_)
    print(f"k={k}: inertia={inertia:.2f}, silhouette={silhouette_avg:.3f}, ARI={ari:.3f}")
    L.append((k, inertia, silhouette_avg, ari))

df_kmeans_eval = pd.DataFrame(L, columns=['k', 'Inertia', 'Silhouette', 'ARI'])
df_kmeans_eval.plot(x='k', y=['Inertia', 'Silhouette', 'ARI'], subplots=True, figsize=(8, 8), title='KMeans Metrics')

kmeans = KMeans(n_clusters=best_k, n_init=15, init='random', max_iter=300, random_state=42)
kmeans.fit(X)
metrics_kmeans = evaluate_clustering("KMeans", kmeans.labels_, X, y_alphabet_test, X_2d)

#=== Bisecting KMeans ===

print("\n--- Bisecting KMeans ---")
bisec = BisectingKMeans(n_clusters=best_k, n_init=10, random_state=42)
bisec.fit(X)
metrics_bisec = evaluate_clustering("Bisecting KMeans", bisec.labels_, X, y_alphabet_test, X_2d)

#=== Gaussian Mixture Model ===

print("\n--- Gaussian Mixture Model ---")
gmm = GaussianMixture(
    n_components=best_k,
    covariance_type='diag',
    reg_covar=1e-3,
    max_iter=500,
    random_state=42
)
gmm_labels = gmm.fit_predict(X)
metrics_gmm = evaluate_clustering("Gaussian Mixture", gmm_labels, X, y_alphabet_test, X_2d)

df_results = pd.DataFrame([
    metrics_kmeans,
    metrics_bisec,
    metrics_gmm
])

#=== Resumo dos Resultados de Clusterização ===

print("\n=== Summary of Clustering Results ===")
print(df_results)
df_results.to_csv("clustering_summary.csv", index=False)
print("Saved clustering_summary.csv")

#=== Busca em Grade para Hiperparâmetros do GMM ===

params_grid = list(product(
    ['diag'],  
    [10, 12, 15],                        
    [1e-6, 1e-3, 1e-2],                      
    [500]                     
))

results = []
for cov_type, n_comp, reg, max_iter in params_grid:
    gmm = GaussianMixture(n_components=n_comp, covariance_type=cov_type, reg_covar=reg, max_iter=max_iter, random_state=42)
    labels = gmm.fit_predict(X)
    if len(set(labels)) > 1:
        sil = silhouette_score(X, labels)
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)
    else:
        sil = ari = nmi = np.nan
    results.append((cov_type, n_comp, reg, sil, ari, nmi))

#=== Resultados da Busca em Grade do GMM ===

df = pd.DataFrame(results, columns=['Covariance', 'Components', 'Reg', 'Silhouette', 'ARI', 'NMI'])
print(df.sort_values('ARI', ascending=False).head(10))
df.to_csv("gmm_hyperparam_tuning.csv", index=False)
print("Saved gmm_hyperparam_tuning.csv")

best_row = df.iloc[0]
best_cov = best_row['Covariance']
best_comp = int(best_row['Components'])
best_reg = float(best_row['Reg'])
print(f"\nBest GMM config -> Covariance: {best_cov}, Components: {best_comp}, Reg: {best_reg}")

#=== Treinamento do Melhor GMM ===

best_gmm = GaussianMixture(
    n_components=best_comp,
    covariance_type=best_cov,
    reg_covar=best_reg,
    random_state=42
)
best_labels = best_gmm.fit_predict(X)


plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=X_2d[:, 0],
    y=X_2d[:, 1],
    hue=best_labels.astype(str),
    palette="tab20",
    s=20,
    alpha=0.8,
    edgecolor=None,
    linewidth=0,
    legend="full"
)
plt.title(f"UMAP Projection (Best GMM Clusters)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.axis("off")
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc=2,
    borderaxespad=0.,
    markerscale=2,
    fontsize='small'
)
plt.tight_layout()
plt.savefig("umap_best_gmm_clusters.png")
plt.close()
print("Saved umap_best_gmm_clusters.png")