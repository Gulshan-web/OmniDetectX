import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "models/authenticity_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
idx_to_class = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def load_authenticity_model():
    global model, idx_to_class

    if model is not None and idx_to_class is not None:
        return model, idx_to_class

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Authenticity model not found at {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    idx_to_class = {
        v: k for k, v in checkpoint["class_to_idx"].items()
    }

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    return model, idx_to_class


def predict_authenticity(image_path):
    loaded_model, loaded_idx_to_class = load_authenticity_model()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = loaded_model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    label = loaded_idx_to_class[predicted.item()]
    confidence_score = round(confidence.item() * 100, 2)

    if label == "FAKE":
        result = "AI Generated / Fake Image"
    else:
        result = "Real Image"

    return result, confidence_score


if __name__ == "__main__":
    image_path = "input/sample2.jpg"

    result, confidence = predict_authenticity(image_path)

    print("\nImage Authenticity Result:")
    print("Image:", image_path)
    print("Result:", result)
    print("Confidence:", confidence, "%")