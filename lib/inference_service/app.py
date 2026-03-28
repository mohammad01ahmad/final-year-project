from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


PROJECT_ROOT = "/Users/Ahmad/UNI-work/year3/FYP/Project"

# Model configuration
@dataclass(frozen=True)
class ModelConfig:
    key: str
    title: str
    model_path: Path
    labels: list[str]
    preprocess: Callable[[bytes], tuple[np.ndarray, np.ndarray]]
    mode: str

# Helper function to decode image bytes
def decode_image_bytes(file_bytes: bytes, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, flags)
    if image is None:
      raise ValueError("Could not decode the uploaded image.")
    return image

# Helper function to encode Grad-CAM image to base64
def encode_base64_image(image: np.ndarray) -> str:
    if image.ndim == 2:
        output = image
    else:
        output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", output)
    if not ok:
        raise ValueError("Failed to encode the generated Grad-CAM image.")
    base64_bytes = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{base64_bytes}"

# Helper functions for Grad-CAM
def find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None:
            continue
        if isinstance(output_shape, list):
            continue
        if len(output_shape) == 4:
            return layer.name
    raise ValueError("No convolutional layer found for Grad-CAM.")
def make_gradcam_heatmap(
    input_tensor: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str,
    pred_index: int | None = None,
) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
def overlay_heatmap(display_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = display_image.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_resized = cv2.resize(
        heatmap_uint8,
        (base.shape[1], base.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base.astype(np.uint8), 1 - alpha, colored, alpha, 0)

def crop_brain_region(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        return image

    scale_x = image.shape[1] / 224
    scale_y = image.shape[0] / 224
    x0 = max(int(x * scale_x), 0)
    y0 = max(int(y * scale_y), 0)
    x1 = min(int((x + w) * scale_x), image.shape[1])
    y1 = min(int((y + h) * scale_y), image.shape[0])
    cropped = image[y0:y1, x0:x1]
    return cropped if cropped.size else image

# Preprocessing functions for diseases
def preprocess_alzheimers(file_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    image = decode_image_bytes(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    display = resized.copy()
    array = tf.keras.applications.mobilenet_v2.preprocess_input(
        resized.astype(np.float32)
    )
    return np.expand_dims(array, axis=0), display
def preprocess_brain_tumor(file_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    image = decode_image_bytes(file_bytes, cv2.IMREAD_COLOR)
    cropped = crop_brain_region(image)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 2, 50, 50)
    colored = cv2.applyColorMap(filtered, cv2.COLORMAP_BONE)
    resized = cv2.resize(colored, (200, 200), interpolation=cv2.INTER_AREA)
    display = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = resized.astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0), display
def preprocess_chest_diseases(file_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    image = decode_image_bytes(file_bytes, cv2.IMREAD_COLOR)
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(gray)
    rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    display = rgb.copy()
    tensor = rgb.astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0), display
def preprocess_tuberculosis(file_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    image = decode_image_bytes(file_bytes, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
    display = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=-1)
    return np.expand_dims(tensor, axis=0), display

# Model registry
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "alzheimers": ModelConfig(
        key="alzheimers",
        title="Alzheimers Detection",
        model_path=PROJECT_ROOT + "/alzheimers/models/trained/alzheimers_model.h5",
        labels=[
            "Mild Demented",
            "Moderate Demented",
            "Non Demented",
            "Very MildDemented",
        ],
        preprocess=preprocess_alzheimers,
        mode="multiclass",
    ),
    "brain-tumor": ModelConfig(
        key="brain-tumor",
        title="Brain tumor Detection",
        model_path=PROJECT_ROOT + "/brain-tumor/models/trained/brain_tumor_MRI_resnet50.keras",
        labels=["glioma", "meningioma", "no_tumor", "pituitary"],
        preprocess=preprocess_brain_tumor,
        mode="multiclass",
    ),
    "chest-diseases": ModelConfig(
        key="chest-diseases",
        title="Chest diseases Detection",
        model_path=PROJECT_ROOT + "/chest-diseases/models/trained/best_chest-diseases.keras",
        labels=["COVID-19", "Normal", "Non-COVID"],
        preprocess=preprocess_chest_diseases,
        mode="multiclass",
    ),
    "tuberculosis": ModelConfig(
        key="tuberculosis",
        title="Tuberculosis Detection",
        model_path=PROJECT_ROOT + "/tuberculosis/models/trained/tuberculosis_X_ray.keras",
        labels=["NORMAL", "TUBERCULOSIS"],
        preprocess=preprocess_tuberculosis,
        mode="binary",
    ),
}

# Load models
@lru_cache(maxsize=None)
def load_model_bundle(key: str) -> tuple[tf.keras.Model, str]:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {key}")

    config = MODEL_REGISTRY[key]
    model_path = Path(config.model_path) if isinstance(config.model_path, str) else config.model_path

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    last_conv_layer_name = find_last_conv_layer(model)
    return model, last_conv_layer_name

# Build probabilities
def build_probabilities(labels: list[str], scores: np.ndarray) -> list[dict[str, float | str]]:
    return [
        {"label": label, "score": float(score)}
        for label, score in zip(labels, scores.tolist(), strict=True)
    ]

# Final function - takes image and model key and returns prediction
async def run_inference(model_key: str, upload: UploadFile) -> JSONResponse:
    if upload.content_type is None or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    try:
        file_bytes = await upload.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        config = MODEL_REGISTRY[model_key]
        input_tensor, display_image = config.preprocess(file_bytes)
        model, last_conv_layer_name = load_model_bundle(model_key)
        predictions = model.predict(input_tensor, verbose=0)

        if config.mode == "binary":
            positive_score = float(predictions[0][0])
            class_scores = np.array([1.0 - positive_score, positive_score], dtype=np.float32)
            pred_index = int(positive_score >= 0.5)
        else:
            class_scores = predictions[0].astype(np.float32)
            pred_index = int(np.argmax(class_scores))

        # 1. Save original activation and switch to linear for better gradients
        original_activation = model.layers[-1].activation
        model.layers[-1].activation = tf.keras.activations.linear

        try:
            heatmap = make_gradcam_heatmap(
                input_tensor=input_tensor,
                model=model,
                last_conv_layer_name=last_conv_layer_name,
                pred_index=pred_index,
            )
        finally:
            # 2. IMPORTANT: Always switch it back to avoid breaking subsequent calls
            model.layers[-1].activation = original_activation

        gradcam = overlay_heatmap(display_image, heatmap)


        payload = {
            "prediction": config.labels[pred_index],
            "probabilities": build_probabilities(config.labels, class_scores),
            "gradcamImage": encode_base64_image(gradcam),
            "model": Path(config.model_path).name if isinstance(config.model_path, str) else config.model_path.name,
            "inputSize": {
                "width": int(input_tensor.shape[2]),
                "height": int(input_tensor.shape[1]),
            },
        }
        return JSONResponse(payload)
    except HTTPException:
        raise
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


app = FastAPI(title="Aether Medical Inference Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/alzheimers")
async def predict_alzheimers(file: UploadFile = File(...)) -> JSONResponse:
    return await run_inference("alzheimers", file)


@app.post("/predict/brain-tumor")
async def predict_brain_tumor(file: UploadFile = File(...)) -> JSONResponse:
    return await run_inference("brain-tumor", file)


@app.post("/predict/chest-diseases")
async def predict_chest_diseases(file: UploadFile = File(...)) -> JSONResponse:
    return await run_inference("chest-diseases", file)


@app.post("/predict/tuberculosis")
async def predict_tuberculosis(file: UploadFile = File(...)) -> JSONResponse:
    return await run_inference("tuberculosis", file)
