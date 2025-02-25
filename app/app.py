import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import zipfile
import tempfile
import numpy as np
import matplotlib.pyplot as plt

# Get absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), "traffic_sign_classifier.pth")

# Debugging: Ensure the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Load trained model
class TrafficSignCNN(torch.nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Class label mapping
class_labels = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
    "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
    "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right", "Double curve",
    "Bumpy road", "Slippery road", "Road narrows on the right", "Road work", "Traffic signals",
    "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead", "Ahead only",
    "Go straight or right", "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing by vehicles over 3.5 metric tons"
]

# Initialize and load the model
model = TrafficSignCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Streamlit UI
st.title("Traffic Sign Classifier")
st.write("Upload a folder of traffic sign images (as a ZIP file), and the model will classify them.")

uploaded_file = st.file_uploader("Upload a ZIP file containing images...", type=["zip"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Recursively find image files in subdirectories
        image_files = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith((".jpg", ".png")):
                    image_files.append(os.path.join(root, file))
        
        st.write(f"Found {len(image_files)} images in the uploaded ZIP file.")

        results = []
        images_to_display = []
        for img_path in image_files:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                top1_prob, top1_class = torch.max(probabilities, 0)
            
            predicted_label = class_labels[top1_class.item()]
            results.append((os.path.basename(img_path), predicted_label, top1_prob.item() * 100.0 * 100.0))
            images_to_display.append((image, predicted_label, top1_prob.item() * 100.0))
        
        st.write("### Predictions:")
        cols = st.columns(5)  # Display images in a grid
        for i, (img, label, confidence) in enumerate(images_to_display):
            with cols[i % 5]:
                st.image(img, caption=f"{label} ({confidence:.2f}%)", use_container_width=True)
                st.write(f"{label} ({confidence:.2f}%)")

    # Reset Button
    if st.button("Reset"):  
        st.experimental_rerun()
