from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os


def create_app():
    app = Flask(__name__)

    # Configure upload settings
    app.config["UPLOAD_FOLDER"] = "uploads/audio"
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {"mp3", "wav", "ogg", "m4a"}

    def allowed_file(filename):
        return (
            "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
        )

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/hello")
    def get_hello():
        return "Hello VOCA Health!!!"

    @app.route("/api/upload-audio", methods=["POST"])
    def upload_audio():
        if "audio" not in request.files:
            return {"error": "No audio file provided"}, 400

        audio_file = request.files["audio"]

        if audio_file.filename == "":
            return {"error": "No selected file"}, 400

        if audio_file and allowed_file(audio_file.filename):
            # Create upload directory if it doesn't exist
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

            filename = secure_filename(audio_file.filename)
            audio_file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return {"message": "File uploaded successfully", "filename": filename}, 200

        return {"error": "Invalid file type"}, 400

    return app
