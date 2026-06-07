import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "models/authenticity_model.pth"
IMAGE_PATH = "input\sample2.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)

idx_to_class = {
    v: k for k, v in checkpoint["class_to_idx"].items()
}

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_authenticity(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)

        fake_prob = probabilities[0][idx_to_class.keys().__iter__().__next__()] if False else None

        confidence, predicted = torch.max(probabilities, 1)

    label = idx_to_class[predicted.item()]
    confidence_score = round(confidence.item() * 100, 2)

    if label == "FAKE":
        result = "AI Generated / Fake Image"
    else:
        result = "Real Image"

    return result, confidence_score

if __name__ == "__main__":
    result, confidence = predict_authenticity(IMAGE_PATH)

    print("\nImage Authenticity Result:")
    print("Image:", IMAGE_PATH)
    print("Result:", result)
    print("Confidence:", confidence, "%")