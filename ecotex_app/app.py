import hashlib
import io
import json
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "ecotex_model.keras")
LABELS_PATH = os.path.join(BASE, "model", "class_names.json")
ALLOWED = {"jpg", "jpeg", "png", "webp"}
THRESHOLD = 0.60  # isse kam confidence par "uncertain" warning

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

TIPS = {
    "cotton": "Recycle into rags, insulation, or new cotton fibre. Keep it clean and dry.",
    "denim": "Send to denim take-back programs, or shred into insulation and new yarn.",
    "polyester": "Can be mechanically or chemically recycled into new polyester fibre. Do not burn.",
    "silk": "Rarely recycled at scale. Best reused, repaired, or donated; scraps suit crafts.",
    "wool": "Recycle into felt, blankets, or insulation. Wool is also compostable if untreated.",
}
INFO = {
    "cotton": "Natural plant fibre. Soft, breathable, absorbent.",
    "denim": "Sturdy twill cotton weave, usually indigo dyed.",
    "polyester": "Synthetic fibre from petroleum. Durable, quick drying.",
    "silk": "Natural protein fibre with a smooth, glossy surface.",
    "wool": "Natural animal fibre. Warm, elastic, often fuzzy.",
}

try:
    with open(LABELS_PATH, encoding="utf-8") as f:
        CLASS_NAMES = json.load(f)
except Exception:
    CLASS_NAMES = ["cotton", "denim", "polyester", "silk", "wool"]

# Home page par dikhne wale numbers. Retrain ke baad yahan update kar dena.
STATS = {"classes": 5, "images": "2,662", "accuracy": "63%", "input": "224 px"}
DIST = {"cotton": 692, "wool": 646, "silk": 590, "polyester": 371, "denim": 363}

model = None
try:
    if os.path.exists(MODEL_PATH):
        from tensorflow import keras  # lazy import: app TF ke baghair bhi khul jati hai

        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded:", MODEL_PATH)
    else:
        print("model/ecotex_model.keras nahi mila -> DEMO MODE")
except Exception as e:  # noqa: BLE001
    print("Model load failed, DEMO MODE:", e)


def predict(img_bytes):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    img = img.resize((224, 224))  # training jaisa hi resize; model ke andar normalization hai
    if model is not None:
        arr = np.expand_dims(np.asarray(img, dtype="float32"), 0)  # 0-255 values
        probs = model.predict(arr, verbose=0)[0]
        return probs, False
    seed = int(hashlib.md5(img_bytes).hexdigest()[:8], 16)  # demo: same image = same result
    return np.random.default_rng(seed).dirichlet(np.ones(len(CLASS_NAMES)) * 0.6), True


@app.route("/")
def home():
    return render_template("home.html", page="home", classes=CLASS_NAMES, stats=STATS, dist=DIST)


@app.route("/about")
def about():
    return render_template("about.html", page="about", classes=CLASS_NAMES, info=INFO)


@app.route("/classify")
def classify():
    return render_template("classify.html", page="classify", demo=model is None)


@app.route("/health")
def health():
    return jsonify(status="ok", demo=model is None, classes=CLASS_NAMES)


@app.route("/predict", methods=["POST"])
def predict_route():
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify(error="No image received. Please choose or capture a photo."), 400
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else "jpg"
    if ext not in ALLOWED:
        return jsonify(error="Unsupported file type. Use JPG, PNG or WEBP."), 400
    try:
        probs, demo = predict(f.read())
    except (UnidentifiedImageError, OSError):
        return jsonify(error="This file is not a valid image."), 400
    order = np.argsort(probs)[::-1][:3]
    top3 = [{"label": CLASS_NAMES[i], "confidence": float(probs[i])} for i in order]
    best = top3[0]
    return jsonify(
        label=best["label"],
        confidence=best["confidence"],
        top3=top3,
        tip=TIPS.get(best["label"], ""),
        info=INFO.get(best["label"], ""),
        uncertain=best["confidence"] < THRESHOLD,
        demo=demo,
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is too large (max 8 MB)."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
