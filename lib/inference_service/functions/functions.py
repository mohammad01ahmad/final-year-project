import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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

# --- THE GRAD-CAM ENGINE (The Intelligence) ---
def generate_gradcam(model: tf.keras.Model, img_array: np.ndarray, class_index: int) -> np.ndarray:
    """
    Identifies the area of interest by calculating gradients of the target class 
    with respect to the last convolutional layer.
    """
    # 1. Find the last 4D convolutional layer automatically
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        # We look for the last layer that outputs a 4D tensor (Batch, H, W, Filters)
        if len(layer.output_shape) == 4:
            last_conv_layer_name = layer.name
            break

    if not last_conv_layer_name:
        return np.zeros(img_array.shape[1:3]) # Fallback

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, class_index]

    # 2. Compute gradients and neuron importance
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 3. Weigh the feature map by the importance
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 4. Normalize: ReLU (only positive influence) and 0-1 scaling
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()
def apply_heatmap(heatmap: np.ndarray, original_img: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlays the heatmap onto the original image with clinical-grade precision.
    """
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Convert to RGB Heatmap (Jet color map)
    heatmap_color = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Superimpose
    output = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return output

# --- PREPROCESSING FACTORY ---
def preprocess_image(file_bytes: bytes, config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    # Decode
    nparr = np.frombuffer(file_bytes, np.uint8)
    decode_flag = cv2.IMREAD_GRAYSCALE if config.color_mode == "grayscale" else cv2.IMREAD_COLOR
    img = cv2.imdecode(nparr, decode_flag)
    if img is None:
        raise ValueError("Invalid image")

    if config.color_mode == "grayscale":
        img_resized = cv2.resize(img, config.target_size)
        input_tensor = np.expand_dims(img_resized, axis=(0, -1))
        display_img = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, config.target_size)
        input_tensor = np.expand_dims(img_resized, axis=0)
        display_img = img_resized

    # Preprocess for model
    if config.preprocess_type == "rescale":
        input_tensor = input_tensor.astype("float32") / 255.0

    return input_tensor, display_img
