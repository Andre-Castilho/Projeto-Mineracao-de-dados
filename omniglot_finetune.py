# === PyTorch e Treinamento ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

# === Visão Computacional ===
from torchvision import transforms, models
from torchvision.datasets import Omniglot
from PIL import Image

# === Utilidades ===
from pathlib import Path
from tqdm import tqdm
from termcolor import colored

# === Machine Learning e Avaliação ===
from sklearn.model_selection import train_test_split

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

train_val_folder = base_path / "images_background"
train_val_dataset = OmniglotSplit(train_val_folder, train_alphabets, transform)

test_folder = base_path / "images_evaluation"
test_dataset = OmniglotSplit(test_folder, test_alphabets, transform)

dataset_indices = list(range(len(train_val_dataset)))
train_indices, val_indices = train_test_split(dataset_indices, test_size=0.2, random_state=42)

train_subset = Subset(train_val_dataset, train_indices)
val_subset = Subset(train_val_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#=== Configuração do Modelo, Critério e Otimizador ===
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes) 
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 100
best_val_accuracy = 0.0
model_save_path = "resnet50.pth"
accuracy_loss_counter = 0

#=== Loop de Treinamento e Validação ===
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        accuracy_loss_counter = 0
        torch.save(model.state_dict(), model_save_path)
        print(colored(f"New best model saved with accuracy: {best_val_accuracy:.2f}%", "green"))
    else:
        accuracy_loss_counter += 1
        print(colored(f"No improvement. Counter: {accuracy_loss_counter}/10", "red"))
        if accuracy_loss_counter >= 10:
            print(colored("Early stopping triggered.", "yellow"))
            break

#=== Finalização ===
print("Finished Fine-Tuning. Best Val Accuracy: {:.2f}%".format(best_val_accuracy))
