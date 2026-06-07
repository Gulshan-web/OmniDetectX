import os

from object_detector import detect_objects_image
from ocr_detector import extract_text
from handwriting_detector import read_handwriting
from authenticity_predict import predict_authenticity
from scene_understanding import understand_scene
from database import create_database, save_scan


def generate_report(image_path):

    create_database()

    print("\n========== OmniDetect X AI ==========\n")
    print("Processing image:", image_path)

    print("\n[1] Object Detection...")
    objects, object_output = detect_objects_image(image_path)

    print("\n[2] Printed Text OCR...")
    printed_text = extract_text(image_path)

    print("\n[3] Handwriting Recognition...")

    object_names_temp = [obj["object"] for obj in objects]

    handwriting_related_objects = [
        "book",
        "notebook",
        "paper",
        "document"
    ]

    should_check_handwriting = any(
        item in object_names_temp
        for item in handwriting_related_objects
    )

    if should_check_handwriting:
        try:
            handwriting_text = read_handwriting(image_path)
            clean_text = handwriting_text.strip()

            if len(clean_text) < 15:
                handwriting_text = "No handwriting detected"

        except Exception:
            handwriting_text = "No handwriting detected"
    else:
        handwriting_text = "No handwriting detected"

    print("\n[4] Scene Understanding...")
    try:
        scene_caption = understand_scene(image_path)
    except Exception:
        scene_caption = "Scene understanding failed"

    print("\n[5] AI vs Real Image Detection...")
    authenticity_result, authenticity_confidence = predict_authenticity(
        image_path
    )

    object_names = [obj["object"] for obj in objects]
    objects_string = ", ".join(object_names)

    save_scan(
        os.path.basename(image_path),
        objects_string,
        authenticity_result,
        authenticity_confidence
    )

    if object_names:
        object_text = ", ".join(object_names)
    else:
        object_text = "no major objects"

    if printed_text:
        text_status = "printed text was detected"
    else:
        text_status = "no printed text was detected"

    if handwriting_text and handwriting_text != "No handwriting detected":
        handwriting_status = "handwritten content was detected"
    else:
        handwriting_status = "no handwriting was detected"

    smart_summary = (
        f"This image contains {object_text}. "
        f"Scene understanding says: {scene_caption}. "
        f"{text_status.capitalize()} and {handwriting_status}. "
        f"The image appears to be {authenticity_result.lower()} "
        f"with {authenticity_confidence}% confidence."
    )

    report = "\n========== FINAL REPORT ==========\n"

    report += "\nObjects Detected:\n"
    if objects:
        for obj in objects:
            report += f"- {obj['object']} ({obj['confidence']}%)\n"
    else:
        report += "- No objects detected\n"

    report += "\nPrinted Text Extracted:\n"
    if printed_text:
        for item in printed_text:
            report += f"- {item['text']} ({item['confidence']}%)\n"
    else:
        report += "- No printed text detected\n"

    report += "\nHandwriting Text:\n"
    report += f"- {handwriting_text}\n"

    report += "\nScene Understanding:\n"
    report += f"- {scene_caption}\n"

    report += "\nSmart Summary:\n"
    report += f"- {smart_summary}\n"

    report += "\nImage Authenticity:\n"
    report += f"- Result: {authenticity_result}\n"
    report += f"- Confidence: {authenticity_confidence}%\n"

    report += "\nOutput Files:\n"
    report += f"- Object detected image: {object_output}\n"
    report += "- Final report: output/final_report.txt\n"

    report += "\n==================================\n"

    os.makedirs("output", exist_ok=True)

    with open(
        "output/final_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    print(report)


if __name__ == "__main__":

    image_path = input("Enter image path: ")

    if not os.path.exists(image_path):
        print("Image not found.")
    else:
        generate_report(image_path)