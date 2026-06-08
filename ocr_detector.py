import cv2
import re

reader = None


def get_reader():
    global reader

    if reader is None:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)

    return reader


def is_garbage_text(text):
    text = text.strip()

    if len(text) < 3:
        return True

    letters = sum(c.isalpha() for c in text)

    if letters == 0:
        return True

    if re.fullmatch(r"[^a-zA-Z]+", text):
        return True

    return False


def run_ocr_on_image(image):
    ocr_reader = get_reader()
    results = ocr_reader.readtext(image)
    texts = []

    for item in results:
        text = item[1].strip()
        confidence = round(item[2] * 100, 2)

        if confidence < 45:
            continue

        if is_garbage_text(text):
            continue

        texts.append({
            "text": text,
            "confidence": confidence
        })

    return texts


def extract_text(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found:", image_path)
        return []

    image = cv2.resize(image, None, fx=1.5, fy=1.5)

    rotations = [
        image,
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    final_texts = []
    seen = set()

    for rotated in rotations:
        texts = run_ocr_on_image(rotated)

        for item in texts:
            key = item["text"].lower()

            if key not in seen:
                seen.add(key)
                final_texts.append(item)

    return final_texts


if __name__ == "__main__":
    image_path = "input/sample.jpg"
    extracted = extract_text(image_path)

    print("\nExtracted Text:\n")

    if not extracted:
        print("No printed text detected")
    else:
        for item in extracted:
            print(item)