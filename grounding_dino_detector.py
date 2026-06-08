import os
import cv2
import torch
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

DEFAULT_OBJECT_PROMPT = (
    "books. papers. bags. plants. people. animals. "
    "bottle. water bottle. pen. pencil. mouse. computer mouse. "
    "gaming mouse. mousepad. desk mat. notebook. tablet. tablet cover. "
    "book. charger. cable. wire. adapter. desk. table. phone. keyboard. "
    "laptop. cup. bag. chair. plant. person. screen. monitor."
)

BLOCKED_LABELS = [
    "scissors",
    "knife",
    "weapon",
    "unknown",
    "object",
    "objects",
    "objects objects stationery",
    "stationery",
    "person"
]

def clean_label(label):
    label = label.lower().strip()
    label = re.sub(r"[^a-zA-Z\s]", "", label)
    label = re.sub(r"\s+", " ", label).strip()

    replacements = {
        "mobile phone": "phone",
        "cell phone": "phone",
        "computer mouse": "mouse",
        "wireless mouse": "mouse",
        "mouse device": "mouse",
        "gaming mouse": "mouse",
        "desk mat": "mousepad",
        "table mat": "mousepad",
        "writing pen": "pen",
        "ball pen": "pen",
        "blue pen": "pen",
        "pencil pen": "pen",
        "spiral notebook": "notebook",
        "copy book": "notebook",
        "tablet cover": "notebook",
        "laptop computer": "laptop",
        "water bottle": "bottle"
    }

    for key, value in replacements.items():
        if key in label:
            return value

    words = label.split()

    priority_words = [
        "bottle", "cup", "pen", "pencil", "mouse", "mousepad",
        "notebook", "book", "phone", "charger", "cable", "wire",
        "tablet", "desk", "table", "chair", "keyboard", "laptop",
        "bag", "plant", "person", "screen", "monitor", "remote"
    ]

    for word in priority_words:
        if word in words:
            return word

    return label


def is_valid_label(label, confidence):
    if not label:
        return False

    if label in BLOCKED_LABELS:
        return False

    if len(label) < 3:
        return False

    if confidence < 35:
        return False

    generic_words = [
        "item",
        "thing",
        "stuff",
        "area",
        "part",
        "visible",
        "objects"
    ]

    if label in generic_words:
        return False

    return True


def detect_open_objects(image_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        text=DEFAULT_OBJECT_PROMPT,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.20,
    text_threshold=0.20,
    target_sizes=[image.size[::-1]]
    )[0]

    cv_image = cv2.imread(image_path)
    best_objects = {}

    for box, score, label in zip(
        results["boxes"],
        results["scores"],
        results["labels"]
    ):
        confidence = round(float(score) * 100, 2)
        clean_name = clean_label(label)

        if not is_valid_label(clean_name, confidence):
            continue

        if (
            clean_name not in best_objects
            or confidence > best_objects[clean_name]["confidence"]
        ):
            best_objects[clean_name] = {
                "object": clean_name,
                "confidence": confidence,
                "box": box
            }

    detected = []

    for name, data in best_objects.items():
        confidence = data["confidence"]
        box = data["box"]

        x1, y1, x2, y2 = map(int, box.tolist())

        detected.append({
            "object": name,
            "confidence": confidence
        })

        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            cv_image,
            f"{name} {confidence}%",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    detected = sorted(
        detected,
        key=lambda x: x["confidence"],
        reverse=True
    )

    output_path = os.path.join(output_dir, "grounding_dino_detected.jpg")
    cv2.imwrite(output_path, cv_image)

    return detected, output_path


if __name__ == "__main__":
    image_path = input("Enter image path: ")

    objects, output_path = detect_open_objects(image_path)

    print("\nGrounding DINO Objects:")
    for obj in objects:
        print(obj)

    print("\nOutput saved at:", output_path)