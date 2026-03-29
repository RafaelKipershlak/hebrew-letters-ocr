import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# ==========================================
# 1. Hardcoded Dictionary 
# ==========================================
CODE_LIST = list(range(1488, 1515))
INDEX_TO_CHAR = {index: chr(code) for index, code in enumerate(CODE_LIST)}

# ==========================================
# 2. Model Architecture (Matches training)
# ==========================================
class HebrewOCRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64*16*16, out_features=len(CODE_LIST))

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

# ==========================================
# 3. Smart Preprocessing (Aspect Ratio + Padding)
# ==========================================
def process_image_keep_aspect_ratio(image_path):
    """
    Resizes the image to fit within 64x64 while maintaining its original aspect ratio.
    Pads the remaining space with white pixels (255) to prevent distortion.
    """
    img = Image.open(image_path).convert('L') # Convert to grayscale
    
    # 1. Resize while keeping aspect ratio
    img.thumbnail((64, 64), Image.Resampling.LANCZOS)
    
    # 2. Create a blank white 64x64 canvas
    background = Image.new('L', (64, 64), 255) 
    
    # 3. Paste the resized image into the center of the canvas
    offset = ((64 - img.width) // 2, (64 - img.height) // 2)
    background.paste(img, offset)
    
    # 4. Convert to Tensor
    tensor_img = T.ToTensor()(background).unsqueeze(0)
    return tensor_img, background

# ==========================================
# 4. Prediction Function
# ==========================================
def predict_image(image_path, model_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found.")
        return
    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights '{model_path}' not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on: {device}...")

    # Load Model
    model = HebrewOCRModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Smart Preprocessing
    img_tensor, img_visual = process_image_keep_aspect_ratio(image_path)
    img_tensor = img_tensor.to(device)

    # Inference with Confidence Score
    with torch.no_grad():
        output = model(img_tensor)
        
        # Calculate Softmax probabilities
        probs = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
        predicted_char = INDEX_TO_CHAR[predicted_idx.item()]
        confidence_score = confidence.item() * 100

    # Print CLI Result
    print("\n" + "="*35)
    print(" ✅ PREDICTION COMPLETE!")
    print(f" 🎯 Predicted Letter: {predicted_char}")
    print(f" 📊 Confidence Score: {confidence_score:.2f}%")
    print("="*35 + "\n")

    # Display Image with Matplotlib
    plt.figure(figsize=(4, 4))
    plt.imshow(img_visual, cmap='gray')
    
    title_text = f"Prediction: {predicted_char}\nConfidence: {confidence_score:.2f}%"
    plt.title(title_text, fontsize=18, color='darkblue')
    plt.axis('off')
    
    # Adds a small subtitle to explain this is the processed input
    plt.suptitle("Model Input (Padded to 64x64)", fontsize=10, color='gray', y=0.05)
    plt.show()

# ==========================================
# 5. CLI Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Hebrew OCR Model on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model", type=str, default="hebrew_ocr_augmented.pth", help="Path to model weights")
    
    args = parser.parse_args()
    predict_image(args.image, args.model)
