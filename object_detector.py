from ultralytics import YOLO
import cv2
import os
from collections import defaultdict

model = None


def get_model():
    global model

    if model is None:
        model = YOLO("yolov8m.pt")

    return model


def detect_objects_image(image_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    yolo_model = get_model()

    results = yolo_model(
        image_path,
        conf=0.50,
        iou=0.35,
        max_det=30,
        imgsz=960
    )

    result = results[0]

    annotated = result.plot()
    output_path = os.path.join(
        output_dir,
        "object_detected.jpg"
    )

    cv2.imwrite(output_path, annotated)

    best_objects = defaultdict(float)

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        name = yolo_model.names[class_id]

        if confidence > best_objects[name]:
            best_objects[name] = confidence

    objects = []

    for name, confidence in best_objects.items():
        objects.append({
            "object": name,
            "confidence": round(
                confidence * 100,
                2
            )
        })

    objects = sorted(
        objects,
        key=lambda x: x["confidence"],
        reverse=True
    )

    return objects, output_path


if __name__ == "__main__":
    image_path = "input/sample.jpg"

    objects, output_path = detect_objects_image(
        image_path
    )

    print("\nObjects Detected:")

    for obj in objects:
        print(obj)

    print(
        "\nOutput saved at:",
        output_path
    )