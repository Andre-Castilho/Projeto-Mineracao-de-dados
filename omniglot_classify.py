# === PyTorch e Dataset ===
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import Omniglot

# === Utilidades e Sistema ===
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import random

# === Machine Learning (Scikit-learn) ===
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# === Visualização e Redução de Dimensionalidade ===
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

print(f"Training on {len(train_alphabets)} alphabets (the 'background' set).")
print(f"Testing on {len(test_alphabets)} alphabets (the 'evaluation' set).")

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#=== Testando Vários Classificadores ===
#=== K-Nearest Neighbors (KNN) ===

print("\n--- Testing K-Nearest Neighbors (KNN) ---")
clf_knn = KNeighborsClassifier(n_neighbors=5) 

param_grid = {'n_neighbors': [5, 10, 15],'weights':['uniform','distance'],'metric':['euclidean','manhattan'], 'leaf_size':[20,30,40]}
best_knn = GridSearchCV(clf_knn, param_grid, scoring=('f1_weighted'), cv=3, refit=True)

scores = cross_validate(best_knn, X, y, cv=3, scoring=('f1_weighted', 'precision_weighted', 'recall_weighted'), return_train_score=True)

print("\n- test -")
print('test_f1_weighted')
print(np.mean(scores['test_f1_weighted']),scores['test_f1_weighted'])

print('test_precision_weighted')
print(np.mean(scores['test_precision_weighted']),scores['test_precision_weighted'])

print('test_recall_weighted')
print(np.mean(scores['test_recall_weighted']),scores['test_recall_weighted'])

print("\n- train -")
print('train_f1_weighted')
print(np.mean(scores['train_f1_weighted']),scores['train_f1_weighted'])

print('train_precision_weighted')
print(np.mean(scores['train_precision_weighted']),scores['train_precision_weighted'])

print('train_recall_weighted')
print(np.mean(scores['train_recall_weighted']),scores['train_recall_weighted'])

model_knn = best_knn.fit(X_train, y_train)

print("\nBest Parameters:", best_knn.best_params_)
print("Best CV Score (f1_weighted):", best_knn.best_score_)

y_true, y_pred_knn = y_test, model_knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"KNN (k={best_knn.best_params_['n_neighbors']}) Accuracy: {knn_accuracy * 100:.2f}%")
print(classification_report(y_true, y_pred_knn))

#=== Random Forest Classifier ===

print("\n--- Testing Random Forest ---")
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)

param_grid = {'n_estimators': [100,200],'criterion':['gini','entropy'], 'max_depth':[5,15,25], 'max_features':['sqrt'], 'random_state':[42]}
best_forest = GridSearchCV(clf_rf, param_grid, scoring=('f1_weighted'), cv=3, refit=True)

scores = cross_validate(best_forest, X, y, cv=3, scoring=('f1_weighted', 'precision_weighted', 'recall_weighted'), return_train_score=True)

print("\n- test -")
print('test_f1_weighted')
print(np.mean(scores['test_f1_weighted']),scores['test_f1_weighted'])

print('test_precision_weighted')
print(np.mean(scores['test_precision_weighted']),scores['test_precision_weighted'])

print('test_recall_weighted')
print(np.mean(scores['test_recall_weighted']),scores['test_recall_weighted'])

print("\n- train -")
print('train_f1_weighted')
print(np.mean(scores['train_f1_weighted']),scores['train_f1_weighted'])

print('train_precision_weighted')
print(np.mean(scores['train_precision_weighted']),scores['train_precision_weighted'])

print('train_recall_weighted')
print(np.mean(scores['train_recall_weighted']),scores['train_recall_weighted'])

model_rf = best_forest.fit(X_train, y_train)

print("\nBest Parameters:", best_forest.best_params_)
print("Best CV Score (f1_weighted):", best_forest.best_score_)

y_true, y_pred_rf = y_test, model_rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print(classification_report(y_true, y_pred_rf))

#=== Logistic Regression (Linear Classifier) ===

print("\n--- Testing Linear Classifier (Logistic Regression) ---")
linear_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
param_grid = {'logisticregression__max_iter': [1000],'logisticregression__solver': ['lbfgs','saga'], 'logisticregression__random_state':[42]}
best_linear = GridSearchCV(linear_pipeline, param_grid, scoring=('f1_weighted'), cv=3, refit=True)

scores = cross_validate(best_linear, X, y, cv=3, scoring=('f1_weighted', 'precision_weighted', 'recall_weighted'), return_train_score=True)

print("\n- test -")
print('test_f1_weighted')
print(np.mean(scores['test_f1_weighted']),scores['test_f1_weighted'])

print('test_precision_weighted')
print(np.mean(scores['test_precision_weighted']),scores['test_precision_weighted'])

print('test_recall_weighted')
print(np.mean(scores['test_recall_weighted']),scores['test_recall_weighted'])

print("\n- train -")
print('train_f1_weighted')
print(np.mean(scores['train_f1_weighted']),scores['train_f1_weighted'])

print('train_precision_weighted')
print(np.mean(scores['train_precision_weighted']),scores['train_precision_weighted'])

print('train_recall_weighted')
print(np.mean(scores['train_recall_weighted']),scores['train_recall_weighted'])

model_linear = best_linear.fit(X_train, y_train)

print("\nBest Parameters:", best_linear.best_params_)
print("Best CV Score (f1_weighted):", best_linear.best_score_)

y_true, y_pred_linear = y_test, model_linear.predict(X_test)

linear_accuracy = accuracy_score(y_test, y_pred_linear)
print(f"Logistic Regression (Linear) Accuracy: {linear_accuracy * 100:.2f}%")
print(classification_report(y_true, y_pred_linear))

#=== Support Vector Classifier (SVC) ===

print("\n--- SVC Classifier ---")
clf_svc = SVC(kernel='rbf', probability=True, random_state=42)
param_grid = {'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'poly'], 'random_state':[42], 'probability':[True]}
best_svc = GridSearchCV(clf_svc, param_grid, scoring=('f1_weighted'), cv=3, refit=True)

scores = cross_validate(best_svc, X, y, cv=3, scoring=('f1_weighted', 'precision_weighted', 'recall_weighted'), return_train_score=True)

print("\n- test -")
print('test_f1_weighted')
print(np.mean(scores['test_f1_weighted']),scores['test_f1_weighted'])

print('test_precision_weighted')
print(np.mean(scores['test_precision_weighted']),scores['test_precision_weighted'])

print('test_recall_weighted')
print(np.mean(scores['test_recall_weighted']),scores['test_recall_weighted'])

print("\n- train -")
print('train_f1_weighted')
print(np.mean(scores['train_f1_weighted']),scores['train_f1_weighted'])

print('train_precision_weighted')
print(np.mean(scores['train_precision_weighted']),scores['train_precision_weighted'])

print('train_recall_weighted')
print(np.mean(scores['train_recall_weighted']),scores['train_recall_weighted'])

model_svc = best_svc.fit(X_train, y_train)

print("\nBest Parameters:", best_svc.best_params_)
print("Best CV Score (f1_weighted):", best_svc.best_score_)

y_true, y_pred_svc = y_test, model_svc.predict(X_test)

svc_accuracy = accuracy_score(y_test, y_pred_svc)
print(f"SVC Accuracy: {svc_accuracy * 100:.2f}%")
print(classification_report(y_true, y_pred_svc))

#=== Resumo dos Resultados dos Classificadores ===

classifier_results = {
    "SVC": (best_svc.best_score_, y_pred_svc),
    "Logistic Regression": (best_linear.best_score_, y_pred_linear),
    "KNN (k=5)": (best_knn.best_score_, y_pred_knn),
    "Random Forest": (best_forest.best_score_, y_pred_rf),
}

best_classifier_name = max(classifier_results, key=lambda k: classifier_results[k][0])
best_f1_weighted, best_predictions = classifier_results[best_classifier_name]

#=== Matriz de Confusão para o Melhor Classificador ===

print(f"\nBest classifier was: {best_classifier_name} with {best_f1_weighted * 100:.2f}% f1_weighted.")
print(f"Generating Confusion Matrix for {best_classifier_name}")
cm = confusion_matrix(y_test, best_predictions, normalize='true') 

plt.figure(figsize=(12, 10))
sns.heatmap(cm * 100, annot=True, fmt=".2f", cmap='Blues', 
            xticklabels=test_alphabets, yticklabels=test_alphabets)
plt.title(f"Confusion Matrix ({best_classifier_name} on Unseen Embeddings)") 
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(f"confusion_matrix_best_classifier.png") 
plt.close() 
print(f"Saved confusion_matrix_best_classifier.png")
