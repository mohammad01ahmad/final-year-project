from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"


def generate_clinical_explanation(
    disease_type: str,
    prediction: str,
    confidence_percent: float,
    location: str,
    context: str,
    model: str = DEFAULT_MODEL,
) -> str:
    if disease_type.casefold() == "tuberculosis":
        system_prompt = "You are a Senior Thoracic Radiologist. You are explaining why an AI model flagged a Chest X-ray."
        contrast_line = (
            "If the prediction is Tuberculosis, identify specific features like opacities or cavities "
            "mentioned in the references that match the AI focus."
        )
    elif disease_type.casefold() == "chest_diseases":
        system_prompt = "You are a Senior Thoracic Radiologist. You are explaining why an AI model flagged a Chest X-ray."
        contrast_line = (
            "Correlate the predicted class with zone-specific thoracic findings such as peripheral or diffuse "
            "opacities, consolidation, and whether the pattern is more consistent with covid-19, non_covid disease, or normal."
        )
    else:
        system_prompt = (
            "You are a Senior Neuroradiologist providing a grounded diagnostic rationale. "
            "Your goal is to validate an AI prediction by mapping spatial heat map data "
            "to specific radiographic semiology found in reference literature."
        )
        contrast_line = (
            "Correlate the predicted tumor class with regional MRI features such as extra-axial or intra-axial "
            "appearance, edema, mass effect, sellar involvement, or posterior fossa findings."
        )

    prompt = f"""
SYSTEM: {system_prompt}
REFERENCE REPORTS:
{context}

NEW CASE DATA:
- Prediction Class: {prediction}
- Confidence: {confidence_percent:.2f}%
- Localization: {location}

INSTRUCTIONS:
1. Direct Correlation: Explain how the visual findings in the {location} correlate with the provided reference reports.
2. Contrast: {contrast_line}
3. Conciseness: Do not repeat yourself. Do not use fluff.
4. Structure:
   - 'Visual Analysis': (Why that location was flagged)
   - 'Clinical Correlation': (How it matches the predicted class)
5. Constraint: Max 90 words. No markdown. No introductory sentences like 'Based on the reports'.
""".strip()

    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
            },
        }
    ).encode("utf-8")

    request = Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(2):
        try:
            with urlopen(request, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except URLError as error:
            if attempt == 1:
                raise RuntimeError(f"Ollama request failed: {error}") from error

    explanation = str(body.get("response", "")).strip()
    if not explanation:
        raise RuntimeError("Ollama returned an empty explanation.")
    return explanation

# TB Explanation 
def generate_tb_explanation(
    prediction: str,
    confidence_percent: float,
    location: str,
    context: str,
    model: str = DEFAULT_MODEL,
) -> str:
    return generate_clinical_explanation(
        disease_type="Tuberculosis",
        prediction=prediction,
        confidence_percent=confidence_percent,
        location=location,
        context=context,
        model=model,
    )

# Brain Tumor Explanation
def generate_brain_tumor_explanation(
    prediction: str,
    confidence_percent: float,
    location: str,
    context: str,
    model: str = DEFAULT_MODEL,
) -> str:
    return generate_clinical_explanation(
        disease_type="Brain_tumor",
        prediction=prediction,
        confidence_percent=confidence_percent,
        location=location,
        context=context,
        model=model,
    )


def generate_chest_diseases_explanation(
    prediction: str,
    confidence_percent: float,
    location: str,
    context: str,
    model: str = DEFAULT_MODEL,
) -> str:
    return generate_clinical_explanation(
        disease_type="Chest_diseases",
        prediction=prediction,
        confidence_percent=confidence_percent,
        location=location,
        context=context,
        model=model,
    )
