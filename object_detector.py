from ultralytics import YOLO
import cv2
import os

# Better accuracy model
model = YOLO("yolov8m.pt")


def detect_objects_image(image_path, output_dir="output"):

    os.makedirs(output_dir, exist_ok=True)

    results = model(
        image_path,
        conf=0.45,     # confidence threshold
        iou=0.30,      # duplicate removal
        max_det=20
    )

    result = results[0]

    annotated = result.plot()

    output_path = os.path.join(
        output_dir,
        "object_detected.jpg"
    )

    cv2.imwrite(output_path, annotated)

    objects = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # extra confidence filter
        if confidence < 0.45:
            continue

        name = model.names[class_id]

        objects.append({
            "object": name,
            "confidence": round(
                confidence * 100,
                2
            )
        })

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