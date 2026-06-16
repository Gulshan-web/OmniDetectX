import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(
    repo_id="gulshan22/omnidetectx-authenticity-model",
    filename="authenticity_model_v2.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
idx_to_class = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_authenticity_model():
    global model, idx_to_class

    if model is not None and idx_to_class is not None:
        return model, idx_to_class

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    idx_to_class = {
        v: k for k, v in checkpoint["class_to_idx"].items()
    }

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    return model, idx_to_class


def predict_authenticity(image_path):
    loaded_model, loaded_idx_to_class = load_authenticity_model()

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = loaded_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    fake_prob = 0
    real_prob = 0

    for idx, prob in enumerate(probabilities):
        label = loaded_idx_to_class[idx].lower()

        if label == "fake":
            fake_prob = prob.item() * 100
        elif label == "real":
            real_prob = prob.item() * 100

    fake_prob = round(fake_prob, 2)
    real_prob = round(real_prob, 2)

    diff = abs(fake_prob - real_prob)

    # Strong decision rules
    if fake_prob >= 75 and diff >= 20:
        result = "AI Generated"
        confidence = fake_prob

    elif real_prob >= 85 and diff >= 30:
        result = "Real"
        confidence = real_prob

    else:
        result = "Uncertain / Needs Manual Check"
        confidence = max(fake_prob, real_prob)

    return result, confidence

if __name__ == "__main__":
    image_path = input("Enter image path: ").strip()

    if not os.path.exists(image_path):
        print("Image not found:", image_path)
    else:
        result, confidence = predict_authenticity(image_path)

        print("\nImage Authenticity Result:")
        print("Image:", image_path)
        print("Result:", result)
        print("Confidence:", confidence, "%")