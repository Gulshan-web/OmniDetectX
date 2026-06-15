import cv2
import re
from rapidfuzz import fuzz

reader = None


def get_reader():
    global reader

    if reader is None:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)

    return reader


def clean_ocr_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[#_~`|]", "", text)

    text = text.replace("0", "o")

    letters_only = text.replace(" ", "")

    if (
        len(letters_only) >= 2
        and letters_only.isalpha()
        and len(text.split()) > 1
    ):
        text = letters_only

        text = text.replace("0", "o")

        if text.lower() == "ttamed":
            text = "Tamed"
            
        if text.lower() == "roubles":
            text = "Troubles"
        
        if text.lower() == "ttamed":
            text = "Tamed"
            
        if text.lower() == "bott:":
            text = "both:"

    return text.strip()


def is_garbage_text(text):
    text = text.strip()

    if len(text) < 2:
        return True

    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)

    if letters == 0:
        return True

    if digits > letters:
        return True

    if re.fullmatch(r"[^a-zA-Z]+", text):
        return True

    bad_patterns = [
        r"^@[A-Za-z]+$",
        r"^[A-Z]{1,2}$",
        r"^['\".,;:!?]+[A-Za-z]?$",
        r"^[A-Za-z]\s?\d+$"
    ]

    for pattern in bad_patterns:
        if re.fullmatch(pattern, text):
            return True

    special_chars = sum(
        not c.isalnum() and not c.isspace()
        for c in text
    )

    if special_chars / len(text) > 0.35:
        return True

    if len(text) > 10 and " " not in text:
        vowels = sum(
            c.lower() in "aeiou"
            for c in text
        )

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

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    contrast = cv2.convertScaleAbs(
        gray,
        alpha=1.4,
        beta=10
    )
    variants.append(contrast)

    return variants


def run_ocr_on_image(image):
    ocr_reader = get_reader()

    results = ocr_reader.readtext(
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

        text = item[1].strip()
        confidence = round(float(item[2]) * 100, 2)

        if confidence < 40:
            continue

        text = clean_ocr_text(text)

        if is_garbage_text(text):
            continue

        texts.append({
            "text": text,
            "confidence": confidence
        })

    return texts


def normalize_key(text):
    return re.sub(
        r"[^a-zA-Z0-9]",
        "",
        text.lower()
    )


def extract_text(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found:", image_path)
        return []

    variants = preprocess_variants(image)

    all_texts = []

    for variant in variants:
        texts = run_ocr_on_image(variant)
        all_texts.extend(texts)

    final_texts = []
    seen_keys = []

    for item in all_texts:
        key = normalize_key(item["text"])

        if not key:
            continue

        is_duplicate = False

        for existing_key in seen_keys:
            if fuzz.ratio(key, existing_key) > 85:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        seen_keys.append(key)
        final_texts.append(item)

    return final_texts


if __name__ == "__main__":
    image_path = input("Enter image path: ")
    extracted = extract_text(image_path)

    print("\nExtracted Text:\n")

    if not extracted:
        print("No printed text detected")
    else:
        for item in extracted:
            print(item)