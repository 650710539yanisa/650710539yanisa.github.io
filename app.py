from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io

from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications import EfficientNetB0, InceptionV3, DenseNet121
from tensorflow.keras import layers

from flask import session, redirect, url_for, flash


app = Flask(__name__)

severity_labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# ---------- สร้างโมเดลแต่ละตัว ----------

def build_efficientnet_model():
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = tf.keras.models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax')
    ])
    return model


def build_inception_model():
    base_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = tf.keras.models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax')
    ])
    return model


def build_densenet_model():
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = tf.keras.models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax')
    ])
    return model


# ---------- โหลดโมเดลและ weight ----------

efficientnet_model = build_efficientnet_model()
efficientnet_model.load_weights(r"C:\DR_project\webapp\models\efficientnetb0_dr_modelsigmoid.h5")

inception_model = build_inception_model()
inception_model.load_weights(r"C:\DR_project\webapp\models\inceptionv3_dr_modelsigmoid.h5")

densenet_model = build_densenet_model()
densenet_model.load_weights(r"C:\DR_project\webapp\models\densenet121_dr_modelsigmoid.h5")

print("✅ All models loaded!")


# ---------- preprocess ใช้กับทุกโมเดล (รูป 224x224, RGB, normalized ด้วย efficientnet preprocess) ----------

def preprocess(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = eff_preprocess(x)
    return x


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read()))
    x = preprocess(img)

    severity_colors = {
        0: "#4CAF50",   # No DR
        1: "#FFEB3B",   # Mild
        2: "#FFC107",   # Moderate
        3: "#FF5722",   # Severe
        4: "#F44336",   # Proliferative DR
    }

    # ---------- รันทั้ง 3 โมเดล ----------
    models_info = [
        ("EfficientNetB0", efficientnet_model),
        ("InceptionV3", inception_model),
        ("DenseNet121", densenet_model),
    ]

    rows_html = ""
    for model_name, mdl in models_info:
        pred = mdl.predict(x)[0]  # shape (5,)
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

    # ---------- สร้าง HTML ตารางไม่มี header / ไม่มีเส้น ----------
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


if __name__ == "__main__":
    app.run(debug=True)