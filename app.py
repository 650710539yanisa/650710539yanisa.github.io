# app.py
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# ถ้าจะดาวน์โหลดจาก Google Drive
try:
    import gdown
except ImportError:
    gdown = None

# ถ้าจะดึงไฟล์ลงมาอยู่ในโฟลเดอร์ 'models'
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# หมายเหตุ: เปลี่ยน ID ให้ตรงกับไฟล์ของคุณ
DRIVE_LINKS = {
    "efficientnet": "https://drive.google.com/uc?id=1usl9QTnNFitQnfSzxOWIztDG-_-GSWGk",
    "inception":   "https://drive.google.com/uc?id=17waZcwMXBRdSPlS6tEb8SenT-qcOLzMI",
    "densenet":    "https://drive.google.com/uc?id=1dLdlUl2bH1tH20a-UW4ovvoJs2KEo2mO",
}

MODEL_FILES = {
    "efficientnet": os.path.join(MODEL_DIR, "efficientnetb0_dr_model.h5"),
    "inception":    os.path.join(MODEL_DIR, "inceptionv3_dr_model.h5"),
    "densenet":     os.path.join(MODEL_DIR, "densenet121_dr_model.h5"),
}

def download_models():
    if not gdown:
        raise RuntimeError("Please install gdown: pip install gdown")
    for name, url in DRIVE_LINKS.items():
        out = MODEL_FILES[name]
        if not os.path.exists(out):
            print(f"📦 Downloading {name} model from Google Drive…")
            gdown.download(url, out, quiet=False)
        else:
            print(f"✅ {name} model already downloaded.")

# ---------- สร้างโมเดลแต่ละตัว ----------
def build_efficientnet_model():
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return model

def build_inception_model():
    base_model = tf.keras.applications.InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return model

def build_densenet_model():
    base_model = tf.keras.applications.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return model

# ---------- โหลดโมเดลและ weight ----------
download_models()

efficientnet_model = build_efficientnet_model()
efficientnet_model.load_weights(MODEL_FILES["efficientnet"])

inception_model = build_inception_model()
inception_model.load_weights(MODEL_FILES["inception"])

densenet_model = build_densenet_model()
densenet_model.load_weights(MODEL_FILES["densenet"])

print("✅ All models loaded!")

# ---------- preprocess ใช้กับทุกโมเดล (รูป 224x224, RGB, normalized) ----------
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as den_preprocess

def preprocess(img, model_name="efficientnet"):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = np.array(img)
    x = np.expand_dims(x, axis=0)

    if model_name == "efficientnet":
        x = eff_preprocess(x)
    elif model_name == "inception":
        x = inc_preprocess(x)
    elif model_name == "densenet":
        x = den_preprocess(x)
    return x

from flask import session, redirect, url_for, flash

app = Flask(__name__)

severity_labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read()))
    x = preprocess(img)

    severity_colors = {
        0: "#4CAF50",
        1: "#FFEB3B",
        2: "#FFC107",
        3: "#FF5722",
        4: "#F44336",
    }

    models_info = [
        ("EfficientNetB0", efficientnet_model),
        ("InceptionV3", inception_model),
        ("DenseNet121", densenet_model),
    ]

    rows_html = ""
    for model_name, mdl in models_info:
        x_pre = preprocess(img, model_name.lower())
        pred = mdl.predict(x_pre)[0]
        pred_class = int(np.argmax(pred))
        confidence = float(np.max(pred))
        color = severity_colors[pred_class]
        severity_text = severity_labels[pred_class]

        rows_html += f"""
        <tr>
            <td style="padding:6px 12px; color:{color}; font-weight:bold;">
                {severity_text}
            </td>
            <td style="padding:6px 12px; font-weight:bold; color:#777777;">
                {model_name}
            </td>
            <td style="padding:6px 12px; font-weight:bold; color:#777777;">
                {confidence:.4f}
            </td>
        </tr>
        """

    result_html = f"""
    <table style="
        width:100%; 
        border-collapse:collapse;
        font-size:14px;
    ">
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """

    return {"result_html": result_html}

# if __name__ == "__main__":
#     app.run(debug=True)
except Exception as e:
        print("❌ Prediction error:", e)  # ดู error จริงใน terminal
        return {"error": str(e)}




