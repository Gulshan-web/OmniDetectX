import cv2
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    show_log=False
)

image_path = r"static\uploads\test2.jpeg"

image = cv2.imread(image_path)

h, w = image.shape[:2]

# Center crop for bottle area
crop = image[
    int(h * 0.25):int(h * 0.85),
    int(w * 0.25):int(w * 0.75)
]

cv2.imwrite("output/bottle_crop.jpg", crop)

results = ocr.ocr("output/bottle_crop.jpg", cls=True)

print("\nPaddleOCR Crop Result:\n")

if not results or not results[0]:
    print("No text detected")
else:
    for line in results[0]:
        text = line[1][0]
        confidence = round(line[1][1] * 100, 2)

        print({
            "text": text,
            "confidence": confidence
        })