from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import shutil

from main import generate_report

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No selected image"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(image_path)

        result = generate_report(image_path)

        detected_image_path = None
        source_detected = "output/object_detected.jpg"
        
        if os.path.exists(source_detected):
            detected_filename = f"detected_{filename}"
            detected_save_path = os.path.join(RESULT_FOLDER, detected_filename)
            shutil.copy(source_detected, detected_save_path)
            detected_image_path = f"/static/results/{detected_filename}"
        
        if os.path.exists(source_detected):
            detected_image_path = os.path.join(RESULT_FOLDER, "object_detected.jpg")
            shutil.copy(source_detected, detected_image_path)

        return render_template(
            "result.html",
            image_path=image_path,
            detected_image_path=detected_image_path,
            objects=result["objects"],
            printed_text=result["printed_text"],
            handwriting=result["handwriting_text"],
            scene=result["scene_caption"],
            authenticity=result["authenticity_result"],
            confidence=result["authenticity_confidence"],
            summary=result["smart_summary"]
            )

    return "Invalid file type"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False, use_reloader=False)