import os

from database import create_database, save_scan


def merge_objects(yolo_objects, open_objects):
    merged_objects = {}

    # 1. YOLO objects always keep
    for obj in yolo_objects:
        name = obj["object"].lower().strip()
        confidence = obj["confidence"]
        merged_objects[name] = confidence

    # 2. Grounding DINO only support/extra objects
    blocked_open_objects = [
        "object",
        "objects",
        "objectsery",
        "tools",
        "stationery",
        "objects objects stationery",
        "unknown",
        "thing",
        "stuff",
        "pad",
        "charge adapter"
    ]

    open_strict_objects = {
        "mouse": 75,
        "mousepad": 75,
        "tie": 90,
        "charger": 55,
        "cable": 55
    }
    

    for obj in open_objects:
        name = obj["object"].lower().strip()
        confidence = obj["confidence"]

        if name in blocked_open_objects:
            continue

        if confidence < 55:
            continue

        if name in open_strict_objects and confidence < open_strict_objects[name]:
            continue

        if name not in merged_objects:
            merged_objects[name] = confidence
        elif confidence > merged_objects[name]:
            merged_objects[name] = confidence

    final_objects = [
        {
            "object": name,
            "confidence": confidence
        }
        for name, confidence in merged_objects.items()
    ]

    return sorted(
        final_objects,
        key=lambda x: x["confidence"],
        reverse=True
    )

def generate_report(image_path):

    create_database()

    print("\n========== OmniDetect X AI ==========\n")
    print("Processing image:", image_path)

    print("\n[1] YOLO Object Detection...")
    try:
        from object_detector import detect_objects_image
        yolo_objects, object_output = detect_objects_image(image_path)
    except Exception as e:
        print("YOLO failed:", e)
        yolo_objects = []
        object_output = "No output image"

    print("\n[1.5] Grounding DINO Open Object Detection...")
    try:
        from grounding_dino_detector import detect_open_objects
        open_objects, open_object_output = detect_open_objects(image_path)
        objects = merge_objects(yolo_objects, open_objects)
    except Exception as e:
        print("Grounding DINO failed:", e)
        objects = yolo_objects
        open_object_output = None

    print("\n[2] Printed Text OCR...", flush=True)
    try:
        from ocr_detector import extract_text
        printed_text = extract_text(image_path)
        print("OCR DEBUG:", printed_text, flush=True)
        print("OCR COUNT:", len(printed_text), flush=True)
    
    except Exception as e:
        print("OCR failed:", e, flush=True)
        printed_text = []

    print("\n[3] Handwriting Recognition...")

    filename_lower = os.path.basename(image_path).lower()

    handwriting_keywords = [
        "handwriting",
        "handwritten",
        "note",
        "notes",
        "paper",
        "assignment",
        "copy",
        "page"
    ]

    should_check_handwriting = any(
        keyword in filename_lower
        for keyword in handwriting_keywords
    )

    if should_check_handwriting:
        try:
            from handwriting_detector import read_handwriting
            handwriting_text = read_handwriting(image_path)
            clean_text = handwriting_text.strip()

            if len(clean_text) < 20:
                handwriting_text = "No handwriting detected"

        except Exception as e:
            print("Handwriting failed:", e)
            handwriting_text = "No handwriting detected"
    else:
        handwriting_text = "No handwriting detected"

    print("\n[4] Florence-2 Scene Understanding...")
    try:
        from florence_understanding import florence_caption
        scene_caption = florence_caption(image_path)

    except Exception as e:
        print("Florence failed:", e)

        try:
            from scene_understanding import understand_scene
            scene_caption = understand_scene(image_path)

        except Exception as e:
            print("BLIP failed:", e)
            scene_caption = "Scene understanding failed"

    if len(scene_caption) > 1000:
        scene_caption = scene_caption[:1000] + "..."

    print("\n[5] AI vs Real Image Detection...")
    try:
        from authenticity_predict import predict_authenticity
        authenticity_result, authenticity_confidence = predict_authenticity(
            image_path
        )
    except Exception as e:
        print("Authenticity failed:", e)
        authenticity_result = "Unknown"
        authenticity_confidence = 0

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

    text_status = (
        "printed text was detected"
        if printed_text
        else "no printed text was detected"
    )

    handwriting_status = (
        "handwritten content was detected"
        if handwriting_text != "No handwriting detected"
        else "no handwriting was detected"
    )

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
    report += f"- YOLO detected image: {object_output}\n"

    if open_object_output:
        report += f"- Grounding DINO image: {open_object_output}\n"

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
    return {
    "objects": objects,
    "printed_text": printed_text,
    "handwriting_text": handwriting_text,
    "scene_caption": scene_caption,
    "authenticity_result": authenticity_result,
    "authenticity_confidence": authenticity_confidence,
    "smart_summary": smart_summary,
    "report": report
    }


if __name__ == "__main__":

    image_path = input("Enter image path: ")

    if not os.path.exists(image_path):
        print("Image not found.")
    else:
        generate_report(image_path)