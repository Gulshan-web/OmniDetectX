import os
import cv2
import re
from rapidfuzz import fuzz

easy_reader = None
paddle_reader = None


def get_easyocr_reader():
    global easy_reader

    if easy_reader is None:
        import easyocr
        easy_reader = easyocr.Reader(["en"], gpu=False)

    return easy_reader


def get_paddle_reader():
    global paddle_reader

    if paddle_reader is None:
        from paddleocr import PaddleOCR

        paddle_reader = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False
        )

    return paddle_reader


def clean_ocr_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[#_~`|]", "", text)
    text = text.replace("0", "o")
    return text.strip()


def is_garbage_text(text):
    text = text.strip()

    if len(text) < 3:
        return True

    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    special_chars = sum(
        not c.isalnum() and not c.isspace()
        for c in text
    )

    if letters == 0:
        return True

    if digits > letters and letters < 3:
        return True

    if len(text) <= 5 and (digits > 0 or special_chars > 0):
        return True

    if special_chars / max(len(text), 1) > 0.45:
        return True

    if len(text) > 12 and " " not in text:
        vowels = sum(c.lower() in "aeiou" for c in text)
        if vowels < 3:
            return True

    return False


def preprocess_variants(image):
    variants = []

    variants.append(image)

    enlarged = cv2.resize(
        image,
        None,
        fx=2.0,
        fy=2.0
    )
    variants.append(enlarged)

    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

    contrast = cv2.convertScaleAbs(
        gray,
        alpha=1.6,
        beta=15
    )
    variants.append(contrast)

    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpen = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    variants.append(sharpen)

    return variants


def run_easyocr_on_image(image):
    reader = get_easyocr_reader()

    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        width_ths=0.7,
        text_threshold=0.4,
        low_text=0.3,
        link_threshold=0.4
    )

    texts = []

    for item in results:
        if len(item) < 3:
            continue

        text = clean_ocr_text(item[1])
        confidence = round(float(item[2]) * 100, 2)

        if confidence < 35:
            continue

        if is_garbage_text(text):
            continue

        texts.append({
            "text": text,
            "confidence": confidence,
            "source": "easyocr"
        })

    return texts


def get_center_crop(image):
    h, w = image.shape[:2]

    return image[
        int(h * 0.25):int(h * 0.85),
        int(w * 0.25):int(w * 0.75)
    ]


def get_right_crop(image):
    h, w = image.shape[:2]

    return image[
        int(h * 0.15):int(h * 0.90),
        int(w * 0.45):int(w * 0.95)
    ]


def run_paddleocr_on_crop(image_path):
    paddle = get_paddle_reader()

    image = cv2.imread(image_path)

    if image is None:
        return []

    os.makedirs("output", exist_ok=True)

    crops = [
        ("center", get_center_crop(image)),
        ("right", get_right_crop(image))
    ]

    texts = []

    for crop_name, crop in crops:
        crop_path = f"output/paddle_{crop_name}_crop.jpg"
        cv2.imwrite(crop_path, crop)

        try:
            results = paddle.ocr(crop_path, cls=True)
        except Exception as e:
            print("PaddleOCR failed:", e)
            continue

        if not results or not results[0]:
            continue

        for line in results[0]:
            text = clean_ocr_text(line[1][0])
            confidence = round(float(line[1][1]) * 100, 2)

            if confidence < 50:
                continue

            if is_garbage_text(text):
                continue

            texts.append({
                "text": text,
                "confidence": confidence,
                "source": "paddleocr"
            })

    return texts


def normalize_key(text):
    return re.sub(
        r"[^a-zA-Z0-9]",
        "",
        text.lower()
    )


def merge_ocr_results(all_texts):
    final_texts = []
    seen = []

    all_texts = sorted(
        all_texts,
        key=lambda x: x["confidence"],
        reverse=True
    )

    for item in all_texts:
        key = normalize_key(item["text"])

        if not key:
            continue

        duplicate = False

        for existing_key in seen:
            if fuzz.ratio(key, existing_key) > 85:
                duplicate = True
                break

        if duplicate:
            continue

        seen.append(key)

        final_texts.append({
            "text": item["text"],
            "confidence": item["confidence"]
        })

    return final_texts


def extract_text(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found:", image_path)
        return []

    all_texts = []

    variants = preprocess_variants(image)

    for variant in variants:
        easy_texts = run_easyocr_on_image(variant)
        all_texts.extend(easy_texts)

    try:
        paddle_texts = run_paddleocr_on_crop(image_path)
        all_texts.extend(paddle_texts)
    except Exception as e:
        print("PaddleOCR fallback failed:", e)

    final_texts = merge_ocr_results(all_texts)

    return final_texts


if __name__ == "__main__":
    image_path = input("Enter image path: ").strip()

    extracted = extract_text(image_path)

    print("\nExtracted Text:\n")

    if not extracted:
        print("No printed text detected")
    else:
        for item in extracted:
            print(item)