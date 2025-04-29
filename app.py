import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# === Define CNN Architecture (MUST match training architecture) ===
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# === Define class labels (adjust based on your dataset folders) ===
class_names = [
    'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral',
    'Contempt', 'Embarrassment', 'Excitement', 'Frustration', 'Amusement',
    'Pain', 'Pride', 'Relief', 'Shame', 'Contentment', 'Confusion', 'Love'
]





# === Load trained model ===``
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=len(class_names)).to(device)

try:
    model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")


# === Define image transformation ===
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# === Streamlit Interface ===
st.title("😊 Emotion Reader")
st.write("Upload a facial image and detect the emotion!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)

    # Predict emotion
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        emotion = class_names[predicted.item()]

    st.success(f"Predicted Emotion: **{emotion.upper()}**")



