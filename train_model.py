import os
import zipfile
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# ==========================================
# 1. Configuration and Paths
# ==========================================
ZIP_PATH = "dataset_archive.zip" # Ensure this file is in the same directory
FOLDER_PATH = "dataset_local"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

# ==========================================
# 2. Data Preparation
# ==========================================
if os.path.exists(ZIP_PATH) and not os.path.exists(FOLDER_PATH):
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(FOLDER_PATH)
    print("Extraction complete.")

def get_number_from_file_name(filename):
    name_without_extenstion = filename.split('.')[0]
    parts = name_without_extenstion.split('_')
    number = parts[-1]
    if number.isdigit():
        return int(number)
    return None

def filenames_to_sorted_unique_list():
    all_filenames = os.listdir(FOLDER_PATH)
    unique_codes = set()
    for file_name in all_filenames:
        if file_name.endswith('.png'):
            code = get_number_from_file_name(file_name)
            if code is not None:
                unique_codes.add(code)
    return sorted(list(unique_codes))

code_list = filenames_to_sorted_unique_list()
code_to_index = {code: index for index, code in enumerate(code_list)}

# ==========================================
# 3. Dataset Construction
# ==========================================
class HebrewLettersDataset(Dataset):
    def __init__(self, folder_path, sorted_codes, transform=None):
        self.folder_path = folder_path
        self.filenames = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        self.transform = transform
        self.code_to_idx = {code: index for index, code in enumerate(sorted_codes)}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert('L') # Convert to grayscale
        
        code = get_number_from_file_name(img_name)
        label = self.code_to_idx[code]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Basic transformation (No Data Augmentation yet)
transform = T.Compose([
    T.ToTensor(),
])

dataset = HebrewLettersDataset(FOLDER_PATH, code_list, transform=transform)

# Split the dataset (80% training, 20% validation)
total_count = len(dataset)
train_count = int(total_count * 0.8)
val_count = total_count - train_count
train_data, val_data = random_split(dataset, [train_count, val_count])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)

# ==========================================
# 4. Model Definition (CNN Architecture)
# ==========================================
class HebrewOCRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64*16*16, out_features=len(code_list))

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

# ==========================================
# 5. Main Execution (Training Loop)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = HebrewOCRModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

    # Validation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Final Validation Accuracy: {accuracy:.2f}%")

    # Save the model weights
    torch.save(model.state_dict(), 'hebrew_ocr_baseline.pth')
    print("Model saved as 'hebrew_ocr_baseline.pth'")
