import easyocr
import cv2
import re

reader = easyocr.Reader(['en'], gpu=False)

def is_garbage_text(text):
    text = text.strip()

    # very short text ignore
    if len(text) < 3:
        return True

    # mostly symbols/numbers ignore
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)

    if letters == 0:
        return True

    # too many random symbols
    if re.fullmatch(r"[^a-zA-Z]+", text):
        return True

    # random short mixed text like Dd, 0o, 79g
    if len(text) <= 4 and digits > 0:
        return True

    return False


def extract_text(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found:", image_path)
        return []

    # resize for better OCR
    image = cv2.resize(image, None, fx=1.5, fy=1.5)

    results = reader.readtext(image)

    texts = []

    for item in results:
        text = item[1].strip()
        confidence = round(item[2] * 100, 2)

        # confidence filter
        if confidence < 70:
            continue

        # garbage filter
        if is_garbage_text(text):
            continue

        texts.append({
            "text": text,
            "confidence": confidence
        })

    return texts


if __name__ == "__main__":
    image_path = "input/sample.jpg"

    extracted = extract_text(image_path)

    print("\nExtracted Text:\n")

    if not extracted:
        print("No printed text detected")
    else:
        for item in extracted:
            print(item)