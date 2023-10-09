import os
import requests
from flask import Flask, request, render_template, redirect, url_for, send_file

app = Flask(__name__)

# Define the endpoint where the video processing will take place
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "flv"}
EXISTING_PROCESSED_VIDEO = "sample.mp4"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            return redirect(request.url)
        
        video_file = request.files["video"]
        
        if video_file.filename == "":
            return redirect(request.url)
        
        if video_file and allowed_file(video_file.filename):
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_file.filename)
            video_file.save(video_path)

            return render_template("result.html", processed_video=EXISTING_PROCESSED_VIDEO)

    return render_template("index.html")

@app.route("/processed_videos/<filename>")
def processed_video(filename):
    return send_file(os.path.join(app.config["PROCESSED_FOLDER"], filename), as_attachment=True)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    app.run(debug=True)