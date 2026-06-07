import cv2
import os
import torch
import easyocr
from PIL import Image, ImageOps, ImageEnhance
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

easy_reader = easyocr.Reader(["en"], gpu=False)

os.makedirs("output/lines", exist_ok=True)

def split_lines(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found:", image_path)
        return None, []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, None, fx=2, fy=2)

    thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (120, 8))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    line_boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if w > 80 and h > 15:
            line_boxes.append((x, y, w, h))

    line_boxes = sorted(line_boxes, key=lambda box: box[1])

    resized_image = cv2.resize(image, None, fx=2, fy=2)

    return resized_image, line_boxes

def preprocess_crop(crop):
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    image = ImageOps.grayscale(image)
    image = ImageEnhance.Contrast(image).enhance(2.5)
    image = image.convert("RGB")
    return image

def read_with_trocr(crop):
    image = preprocess_crop(crop)

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    with torch.no_grad():
        generated_ids = trocr_model.generate(
            pixel_values,
            max_new_tokens=80
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return text

def read_with_easyocr(crop_path):
    results = easy_reader.readtext(crop_path, detail=0)
    return " ".join(results)

def read_handwriting(image_path):
    image, line_boxes = split_lines(image_path)

    if image is None:
        return ""

    final_lines = []

    if len(line_boxes) == 0:
        return "No handwriting detected. Try clearer image."

    for i, (x, y, w, h) in enumerate(line_boxes):
        padding = 20

        crop = image[
            max(0, y-padding): y+h+padding,
            max(0, x-padding): x+w+padding
        ]

        crop_path = f"output/lines/line_{i+1}.jpg"
        cv2.imwrite(crop_path, crop)

        trocr_text = read_with_trocr(crop)
        easy_text = read_with_easyocr(crop_path)

        if len(easy_text.strip()) > len(trocr_text.strip()):
            final_lines.append(easy_text)
        else:
            final_lines.append(trocr_text)

    return "\n".join(final_lines)

if __name__ == "__main__":
    image_path = "input\handwritting.jpg"

    result = read_handwriting(image_path)

    print("\nHandwriting Text:\n")
    print(result)

    print("\nCropped lines saved in output/lines folder.")