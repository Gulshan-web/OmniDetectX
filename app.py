from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import shutil
import uuid
import threading

from main import generate_report

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

jobs = {}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image_job(job_id, image_path, filename):
    try:
        jobs[job_id]["status"] = "processing"

        result = generate_report(image_path)

        detected_image_path = None
        source_detected = "output/object_detected.jpg"

        if os.path.exists(source_detected):
            detected_filename = f"detected_{job_id}_{filename}"
            detected_save_path = os.path.join(RESULT_FOLDER, detected_filename)
            shutil.copy(source_detected, detected_save_path)
            detected_image_path = f"/static/results/{detected_filename}"

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "image_path": image_path,
            "detected_image_path": detected_image_path,
            "objects": result["objects"],
            "printed_text": result["printed_text"],
            "handwriting": result["handwriting_text"],
            "scene": result["scene_caption"],
            "authenticity": result["authenticity_result"],
            "confidence": result["authenticity_confidence"],
            "summary": result["smart_summary"]
        }

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


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

    if not allowed_file(file.filename):
        return "Invalid file type"

    filename = secure_filename(file.filename)
    job_id = str(uuid.uuid4())

    saved_filename = f"{job_id}_{filename}"
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)

    file.save(image_path)

    jobs[job_id] = {
        "status": "queued",
        "result": None,
        "error": None
    }

    thread = threading.Thread(
        target=process_image_job,
        args=(job_id, image_path, filename)
    )
    thread.daemon = True
    thread.start()

    return redirect(url_for("status_page", job_id=job_id))


@app.route("/status/<job_id>")
def status_page(job_id):
    if job_id not in jobs:
        return "Invalid job ID"

    return render_template("status.html", job_id=job_id)


@app.route("/job-status/<job_id>")
def job_status(job_id):
    if job_id not in jobs:
        return {
            "status": "not_found"
        }

    return {
        "status": jobs[job_id]["status"],
        "error": jobs[job_id].get("error")
    }


@app.route("/result/<job_id>")
def result_page(job_id):
    if job_id not in jobs:
        return "Invalid job ID"

    if jobs[job_id]["status"] != "done":
        return redirect(url_for("status_page", job_id=job_id))

    result = jobs[job_id]["result"]

    return render_template(
        "result.html",
        image_path=result["image_path"],
        detected_image_path=result["detected_image_path"],
        objects=result["objects"],
        printed_text=result["printed_text"],
        handwriting=result["handwriting"],
        scene=result["scene"],
        authenticity=result["authenticity"],
        confidence=result["confidence"],
        summary=result["summary"]
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False, use_reloader=False)