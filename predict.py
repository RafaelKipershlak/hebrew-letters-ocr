import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
# 3. Smart Preprocessing (OpenCV Auto-Crop & Binarization)
# ==========================================
def process_real_world_image_cv2(image_path):
    """
    Advanced Preprocessing for real-world images:
    1. Grayscale & Dynamic Binarization.
    2. Contour detection to find and crop the actual ink (Auto-Crop).
    3. Resizes to font scale and pads centrally on a 64x64 white canvas.
    """
    # 1. Read image in Grayscale using OpenCV
    img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")

    # 2. Binarize (Inverted: Background=0, Ink=255 for contour detection)
    _, thresh = cv2.threshold(img_cv, 128, 255, cv2.THRESH_BINARY_INV)

    # 3. Find Contours (Blobs of ink)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 4. Find the LARGEST contour (ignoring tiny noise dots)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop exactly around the largest ink blob (revert to normal: Black ink on White)
        letter_crop = img_cv[y:y+h, x:x+w]
        
        # Ensure binarization of the crop (White background, Black ink)
        _, final_crop = cv2.threshold(letter_crop, 128, 255, cv2.THRESH_BINARY)
    else:
        # Fallback if somehow no ink is found
        _, final_crop = cv2.threshold(img_cv, 128, 255, cv2.THRESH_BINARY)

    # 5. Convert to PIL Image for Resizing and Padding
    pil_img = Image.fromarray(final_crop)
    
    # Resize to font scale (leave some white margin)
    desired_font_scale = 44 
    pil_img.thumbnail((desired_font_scale, desired_font_scale), Image.Resampling.LANCZOS)
    
    # Paste on 64x64 white canvas
    canvas = Image.new('L', (64, 64), 255)
    offset = ((64 - pil_img.width) // 2, (64 - pil_img.height) // 2)
    canvas.paste(pil_img, offset)
    
    tensor_img = T.ToTensor()(canvas).unsqueeze(0)
    return tensor_img, canvas

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

    # Smart Preprocessing (OpenCV)
    try:
        img_tensor, img_visual = process_real_world_image_cv2(image_path)
    except Exception as e:
        print(f"❌ Preprocessing Error: {e}")
        return
        
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
    
    # Updated subtitle to match the new processing
    plt.suptitle("Model Input (Binarized & Auto-Cropped)", fontsize=10, color='gray', y=0.05)
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
