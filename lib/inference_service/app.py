# ------------------------------------
# MAIN ENTRY POINT OF THE APPLICATION: 
# ------------------------------------
# uvicorn app:app --reload

from __future__ import annotations

# App imports 
import base64
import os
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
# Force Legacy Keras if needed for older saved models
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# RAG imports
from RAG import (
    ensure_brain_tumor_vector_store,
    ensure_chest_diseases_vector_store,
    ensure_tb_vector_store,
    extract_brain_tumor_location_from_heatmap,
    extract_chest_location_from_heatmap,
    extract_tb_location_from_heatmap,
    generate_brain_tumor_explanation,
    generate_chest_diseases_explanation,
    generate_tb_explanation,
    retrieve_brain_tumor_context,
    retrieve_chest_diseases_context,
    retrieve_tb_context,
)

# Functions imports
from functions.functions import generate_gradcam, apply_heatmap, preprocess_image

# LLM API imports
from tb_llm_api_experiment import (build_reference_context,generate_llm_api_explanation,select_reference_reports)

# --- CONFIGURATION & PATHS ---
PROJECT_ROOT = Path("/Users/Ahmad/UNI-work/year3/FYP/Project")

@dataclass(frozen=True)
class ModelConfig:
    key: str
    model_path: Path
    labels: list[str]
    target_size: tuple[int, int]
    color_mode: str
    preprocess_type: str  # "rescale" (0-1) or "none" (0-255)

# --- MODEL REGISTRY ---
MODELS_CONFIG = {
    "alzheimers": ModelConfig(
        key="alzheimers",
        model_path=PROJECT_ROOT / "alzheimers/models/trained/best_alzheimers2.keras",
        labels=["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"],
        target_size=(224, 224),
        color_mode="rgb",
        preprocess_type="rescale"
    ),
    "brain-tumor": ModelConfig(
        key="brain-tumor",
        model_path=PROJECT_ROOT / "brain-tumor/models/trained/brain_tumor_MRI_resnet50.keras",
        labels=["glioma", "meningioma", "no_tumor", "pituitary"],
        target_size=(200, 200),
        color_mode="rgb",
        preprocess_type="rescale"
    ),
    "chest-diseases": ModelConfig(
        key="chest-diseases",
        model_path=PROJECT_ROOT / "chest-diseases/models/trained/best_chest-diseases2.keras",
        labels=["COVID-19", "Normal", "Non-COVID"],
        target_size=(224, 224),
        color_mode="rgb",
        preprocess_type="rescale"
    ),
    "tuberculosis": ModelConfig(
        key="tuberculosis",
        model_path=PROJECT_ROOT / "tuberculosis/models/trained/tuberculosis_X_ray_best.keras",
        labels=["NORMAL", "TUBERCULOSIS"],
        target_size=(500, 500),
        color_mode="grayscale",
        preprocess_type="rescale"
    ),
}

@dataclass
class LoadedModel:
    model: tf.keras.Model
    config: ModelConfig

# Cache models in memory for speed
_LOADED_MODELS: dict[str, LoadedModel] = {}

def get_model(key: str) -> LoadedModel:
    if key not in _LOADED_MODELS:
        config = MODELS_CONFIG[key]
        if not config.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {config.model_path}")
        model = tf.keras.models.load_model(str(config.model_path))
        _LOADED_MODELS[key] = LoadedModel(model=model, config=config)
    return _LOADED_MODELS[key]


def normalize_kb_class_label(model_key: str, prediction_label: str) -> str:
    if model_key == "tuberculosis":
        return prediction_label.lower()
    if model_key == "chest-diseases":
        mapping = {
            "COVID-19": "covid-19",
            "Normal": "normal",
            "Non-COVID": "non_covid",
        }
        return mapping.get(prediction_label, prediction_label.lower())
    return prediction_label

# --- API ROUTES ---

app = FastAPI(title="Aether Medical Advanced Inference")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def run_inference_logic(key: str, file: UploadFile):
    try:
        # ---------------
        # 0. Load model
        # ---------------
        loaded = get_model(key)
        content = await file.read()
        
        # ---------------
        # 1. Preprocess
        # ---------------
        input_tensor, original_resized = preprocess_image(content, loaded.config)
        
        # ---------------
        # 2. Prediction
        # ---------------
        preds = loaded.model.predict(input_tensor)
        if loaded.config.key == "tuberculosis":
            tb_score = float(preds[0][0])
            class_idx = 1 if tb_score >= 0.5 else 0
            probability_scores = [1.0 - tb_score, tb_score]
            gradcam_target_index = 0
        else:
            class_idx = int(np.argmax(preds[0]))
            probability_scores = [float(score) for score in preds[0]]
            gradcam_target_index = class_idx
        
        # --------------- 
        # 3. Grad-CAM 
        # --------------- 
        heatmap = generate_gradcam(loaded.model, input_tensor, gradcam_target_index)
        gradcam_img = apply_heatmap(heatmap, original_resized)

        # --------------------------------
        # 4. Prediction Label & Confidence
        # --------------------------------
        prediction_label = loaded.config.labels[class_idx]
        kb_prediction_label = normalize_kb_class_label(loaded.config.key, prediction_label)
        confidence_percent = max(probability_scores) * 100.0
        explanation = None
        rag_summary = None
        rag_references = None
        llm_api_explanation = None
        llm_api_summary = None

        # -------------------------------------
        # 5.1 RAG & LLM API
        # -------------------------------------
        if loaded.config.key == "tuberculosis":
            disease_type = "Tuberculosis"
            location = extract_tb_location_from_heatmap(heatmap)
            retrieved_count = 0
            try:
                ensure_tb_vector_store()
                context, matches = retrieve_tb_context(
                    disease_type=disease_type,
                    prediction=kb_prediction_label,
                    confidence_percent=confidence_percent,
                    location=location,
                )
                retrieved_count = len(matches)
                rag_references = context.strip() or None
                if context.strip():
                    explanation = generate_tb_explanation(
                        prediction=prediction_label,
                        confidence_percent=confidence_percent,
                        location=location,
                        context=context,
                    )
            except Exception as rag_error:
                print(f"TB RAG warning: {rag_error}")

            rag_summary = {
                "location": location,
                "retrievedCount": retrieved_count,
            }

            try:
                simple_reports = select_reference_reports(
                    disease_key="tuberculosis",
                    disease_type=disease_type,
                    prediction=kb_prediction_label,
                    location=location,
                )
                simple_context = build_reference_context(simple_reports)
                if simple_context.strip():
                    llm_api_explanation = generate_llm_api_explanation(
                        disease_key="tuberculosis",
                        prediction=prediction_label,
                        confidence_percent=confidence_percent,
                        location=location,
                        context=simple_context,
                    )
                llm_api_summary = {
                    "location": location,
                    "selectedCount": len(simple_reports),
                }
            except Exception as llm_error:
                print(f"TB LLM API warning: {llm_error}")

        if loaded.config.key == "brain-tumor":
            disease_type = "Brain_tumor"
            location = extract_brain_tumor_location_from_heatmap(heatmap, prediction_label)
            retrieved_count = 0
            try:
                ensure_brain_tumor_vector_store()
                context, matches = retrieve_brain_tumor_context(
                    disease_type=disease_type,
                    prediction=kb_prediction_label,
                    confidence_percent=confidence_percent,
                    location=location,
                )
                retrieved_count = len(matches)
                rag_references = context.strip() or None
                if context.strip():
                    explanation = generate_brain_tumor_explanation(
                        prediction=prediction_label,
                        confidence_percent=confidence_percent,
                        location=location,
                        context=context,
                    )
            except Exception as rag_error:
                print(f"Brain tumor RAG warning: {rag_error}")

            rag_summary = {
                "location": location,
                "retrievedCount": retrieved_count,
            }

            try:
                simple_reports = select_reference_reports(
                    disease_key="brain-tumor",
                    disease_type=disease_type,
                    prediction=kb_prediction_label,
                    location=location,
                )
                simple_context = build_reference_context(simple_reports)
                if simple_context.strip():
                    llm_api_explanation = generate_llm_api_explanation(
                        disease_key="brain-tumor",
                        prediction=prediction_label,
                        confidence_percent=confidence_percent,
                        location=location,
                        context=simple_context,
                    )
                llm_api_summary = {
                    "location": location,
                    "selectedCount": len(simple_reports),
                }
            except Exception as llm_error:
                print(f"Brain tumor LLM API warning: {llm_error}")

        if loaded.config.key == "chest-diseases":
            disease_type = "Chest_diseases"
            location = extract_chest_location_from_heatmap(heatmap)
            retrieved_count = 0
            try:
                ensure_chest_diseases_vector_store()
                context, matches = retrieve_chest_diseases_context(
                    disease_type=disease_type,
                    prediction=kb_prediction_label,
                    confidence_percent=confidence_percent,
                    location=location,
                )
                retrieved_count = len(matches)
                rag_references = context.strip() or None
                if context.strip():
                    explanation = generate_chest_diseases_explanation(
                        prediction=prediction_label,
                        confidence_percent=confidence_percent,
                        location=location,
                        context=context,
                    )
            except Exception as rag_error:
                print(f"Chest diseases RAG warning: {rag_error}")

            rag_summary = {
                "location": location,
                "retrievedCount": retrieved_count,
            }

            try:
                simple_reports = select_reference_reports(
                    disease_key="chest-diseases",
                    disease_type=disease_type,
                    prediction=kb_prediction_label,
                    location=location,
                )
                simple_context = build_reference_context(simple_reports)
                if simple_context.strip():
                    llm_api_explanation = generate_llm_api_explanation(
                        disease_key="chest-diseases",
                        prediction=prediction_label,
                        confidence_percent=confidence_percent,
                        location=location,
                        context=simple_context,
                    )
                llm_api_summary = {
                    "location": location,
                    "selectedCount": len(simple_reports),
                }
            except Exception as llm_error:
                print(f"Chest diseases LLM API warning: {llm_error}")
        
        # 4. Encode to Base64
        _, buffer = cv2.imencode(".png", cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))
        b64_str = base64.b64encode(buffer).decode("utf-8")
        
        return JSONResponse({
            "prediction": loaded.config.labels[class_idx],
            "probabilities": [
                {"label": label, "score": float(score)} 
                for label, score in zip(loaded.config.labels, probability_scores)
            ],
            "gradcamImage": f"data:image/png;base64,{b64_str}",
            "model": loaded.config.key,
            "explanation": explanation,
            "ragSummary": rag_summary,
            "ragReferences": rag_references,
            "llmApiExplanation": llm_api_explanation,
            "llmApiSummary": llm_api_summary,
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/alzheimers")
async def predict_alz (file: UploadFile = File(...)): return await run_inference_logic("alzheimers", file)

@app.post("/predict/brain-tumor")
async def predict_bt (file: UploadFile = File(...)): return await run_inference_logic("brain-tumor", file)

@app.post("/predict/chest-diseases")
async def predict_cd (file: UploadFile = File(...)): return await run_inference_logic("chest-diseases", file)

@app.post("/predict/tuberculosis")
async def predict_tb (file: UploadFile = File(...)): return await run_inference_logic("tuberculosis", file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
