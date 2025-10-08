import os
from typing import Tuple

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Lazy import TensorFlow/Keras to reduce import time for simple GET /
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
from PIL import Image


BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}


def ensure_upload_folder_exists() -> None:
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model():
    # Cache the model on the app object to avoid reloading
    if not hasattr(get_model, "_model"):
        model_path = os.path.join(os.path.dirname(__file__), "tb_model.h5")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        get_model._model = load_model(model_path)  # type: ignore[attr-defined]
    return get_model._model  # type: ignore[attr-defined]


def infer_model_input_size() -> Tuple[int, int, int]:
    model = get_model()
    input_shape = getattr(model, "input_shape", None)
    if input_shape is None:
        # Fallback to a common default if model does not expose input_shape
        return (224, 224, 3)

    # input_shape can be (None, H, W, C) or (H, W, C)
    if len(input_shape) == 4:
        _, height, width, channels = input_shape
    elif len(input_shape) == 3:
        height, width, channels = input_shape
    else:
        # Unexpected; fallback
        return (224, 224, 3)

    # Handle None dimensions by falling back to 224
    height = int(height) if height is not None else 224
    width = int(width) if width is not None else 224
    channels = int(channels) if channels is not None else 3
    if channels not in (1, 3):
        channels = 3
    return (height, width, channels)


def preprocess_image(image_path: str) -> np.ndarray:
    target_h, target_w, channels = infer_model_input_size()

    with Image.open(image_path) as img:
        if channels == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img = img.resize((target_w, target_h))
        img_array = np.array(img).astype("float32")

        # --- NEW: sample-wise normalization ---
        img_array -= np.mean(img_array)
        img_array /= (np.std(img_array) + 1e-7)
        # --- END ---

        if channels == 1 and img_array.ndim == 2:
            img_array = np.expand_dims(img_array, axis=-1)

        img_array = np.expand_dims(img_array, axis=0)
        return img_array



def predict_label(image_path: str) -> Tuple[str, float]:
    model = get_model()
    x = preprocess_image(image_path)
    preds = model.predict(x)

    # Handle common output shapes:
    # 1) Sigmoid binary: shape (1, 1) => value >= 0.5 is TB
    # 2) Softmax 2 classes: shape (1, 2) => argmax 1 is TB (assuming [Normal, TB])
    # 3) Softmax 2 classes reversed: if unclear, use highest prob as TB if its index is 1
    # We will try to infer by shape and choose a sensible default.
    if preds.ndim == 2 and preds.shape[0] == 1:
        if preds.shape[1] == 1:
            prob_tb = float(preds[0, 0])
            is_tb = prob_tb >= 0.5
        elif preds.shape[1] == 2:
            # Assume class order [Normal, TB]
            prob_tb = float(preds[0, 1])
            is_tb = np.argmax(preds[0]) == 1
        else:
            # Unexpected number of classes; use max as TB probability for messaging
            prob_tb = float(np.max(preds[0]))
            is_tb = np.argmax(preds[0]) != 0
    else:
        # Fallback
        prob_tb = float(np.max(preds))
        is_tb = prob_tb >= 0.5

    label = "This is a case of TB" if is_tb else "This X-ray is Normal"
    return label, prob_tb


def create_app() -> Flask:
    ensure_upload_folder_exists()
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            if "file" not in request.files:
                flash("No file part in the request")
                return redirect(request.url)

            file = request.files["file"]
            if file.filename == "":
                flash("No file selected")
                return redirect(request.url)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(save_path)

                try:
                    label, prob_tb = predict_label(save_path)
                except Exception as e:
                    flash(f"Error during prediction: {e}")
                    return redirect(request.url)

                return render_template(
                    "index.html",
                    prediction=label,
                    probability=f"{prob_tb:.3f}",
                    filename=filename,
                )
            else:
                flash("Invalid file type. Please upload a PNG/JPG/JPEG/BMP image.")
                return redirect(request.url)

        return render_template("index.html")

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


