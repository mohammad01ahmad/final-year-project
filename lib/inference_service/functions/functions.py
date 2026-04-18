import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Force Legacy Keras if needed for older saved models
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf


@dataclass(frozen=True)
class ModelConfig:
    key: str
    model_path: Path
    labels: list[str]
    target_size: tuple[int, int]
    color_mode: str  # "rgb" or "grayscale"
    preprocess_type: str  # "rescale" (0-1) or "none" (0-255)


def generate_gradcam(model: tf.keras.Model, img_array: np.ndarray, class_index: int) -> np.ndarray:
    """
    Identifies the area of interest by calculating gradients of the target class
    with respect to the last convolutional layer.
    """
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer_name = layer.name
            break

    if not last_conv_layer_name:
        return np.zeros(img_array.shape[1:3])

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def apply_heatmap(heatmap: np.ndarray, original_img: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlays the heatmap onto the original image with clinical-grade precision.
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_color = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)


def _decode_image(file_bytes: bytes, flag: int) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, flag)
    if img is None:
        raise ValueError("Invalid image")
    return img


def _medical_clahe_preprocess(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return rgb.astype("float32") / 255.0


def _crop_brain_region(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    cropped = img_bgr[y : y + h, x : x + w]
    return cropped if cropped.size else img_bgr


def _prepare_alzheimers(file_bytes: bytes, config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    img_bgr = _decode_image(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, config.target_size)
    processed = img_resized.astype("float32") / 255.0
    return np.expand_dims(processed, axis=0), img_resized


def _prepare_tuberculosis(file_bytes: bytes, config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    gray = _decode_image(file_bytes, cv2.IMREAD_GRAYSCALE)
    gray_resized = cv2.resize(gray, config.target_size)
    processed = gray_resized.astype("float32") / 255.0
    input_tensor = np.expand_dims(processed, axis=(0, -1))
    display_img = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2RGB)
    return input_tensor, display_img


def _prepare_brain_tumor(file_bytes: bytes, config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    img_bgr = _decode_image(file_bytes, cv2.IMREAD_COLOR)
    cropped = _crop_brain_region(img_bgr)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 2, 50, 50)
    pseudocolor = cv2.applyColorMap(filtered, cv2.COLORMAP_BONE)
    pseudocolor_rgb = cv2.cvtColor(pseudocolor, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(pseudocolor_rgb, config.target_size)
    processed = resized.astype("float32") / 255.0
    return np.expand_dims(processed, axis=0), resized


def _prepare_chest_diseases(file_bytes: bytes, config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    img_bgr = _decode_image(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, config.target_size)
    processed = _medical_clahe_preprocess(img_resized)
    display_img = np.clip(processed * 255.0, 0, 255).astype("uint8")
    return np.expand_dims(processed, axis=0), display_img


def preprocess_image(file_bytes: bytes, config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.key == "brain-tumor":
        return _prepare_brain_tumor(file_bytes, config)
    if config.key == "chest-diseases":
        return _prepare_chest_diseases(file_bytes, config)
    if config.key == "tuberculosis":
        return _prepare_tuberculosis(file_bytes, config)
    return _prepare_alzheimers(file_bytes, config)
