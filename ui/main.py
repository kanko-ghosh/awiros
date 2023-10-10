import os
import requests
from flask import Flask, request, render_template, redirect, url_for, send_file

from model import *

app = Flask(__name__)

hi = HelmetIO(yolo_model_path="./models/debas_is_the_goat_model.pt",full_body_yolo_model_path ="./models/yolov8m.pt", classif_model_path="./models/hel0_model.h5", mob_classif_model_path="models/model16.pt")


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
            # video_file.save(video_path)

            hi.convert_v_to_v(src_path="./uploads/"+video_file.filename, dest_path="./processed/"+video_file.filename.split(".")[0]+".gif", read_fps=1.0)
            print(video_file.filename)
            return render_template("result.html", processed_video=video_file.filename.split(".")[0]+".gif")

    return render_template("index.html")

@app.route("/processed_videos/<filename>")
def processed_video(filename):
    print("Sending file: " + filename)
    fullname = os.path.join(app.config["PROCESSED_FOLDER"], filename)
    if (not os.path.isfile(fullname)):
        print("Heinji?!")
    return send_file(os.path.join(app.config["PROCESSED_FOLDER"], filename), as_attachment=True)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    app.run(debug=True)