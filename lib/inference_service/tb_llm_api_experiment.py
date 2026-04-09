from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path("/Users/Ahmad/UNI-work/year3/FYP/Project")
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"

CSV_PATHS = {
    "tuberculosis": PROJECT_ROOT / "fyp/lib/inference_service/RAG/tb_normal.csv",
    "brain-tumor": PROJECT_ROOT / "fyp/lib/inference_service/RAG/brain_tumor.csv",
}


@dataclass(frozen=True)
class SimpleReferenceReport:
    disease_type: str
    class_label: str
    location: str
    findings: str
    impression: str
    score: int


def _normalize(text: str | None) -> str:
    return " ".join((text or "").replace("XXXX", "unspecified").split())


def _get_csv_path(disease_key: str) -> Path:
    if disease_key not in CSV_PATHS:
        raise ValueError(f"Unsupported LLM API experiment disease key: {disease_key}")
    return CSV_PATHS[disease_key]


def _load_reports(disease_key: str) -> list[dict[str, str]]:
    csv_path = _get_csv_path(disease_key)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV knowledge base not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            {
                "disease_type": _normalize(row.get("disease_type")),
                "class_label": _normalize(row.get("class_label")),
                "location": _normalize(row.get("location")),
                "findings": _normalize(row.get("findings")),
                "impression": _normalize(row.get("impression")),
            }
            for row in reader
        ]


def _keywords_for_prediction(disease_key: str, prediction: str) -> list[str]:
    if disease_key == "tuberculosis":
        if prediction.upper() == "TUBERCULOSIS":
            return [
                "tuberculosis",
                "active",
                "cavitary",
                "granuloma",
                "scarring",
                "infiltrate",
                "upper lobe",
                "apical",
                "opacity",
            ]
        return [
            "normal",
            "no active",
            "no evidence",
            "clear lungs",
            "no acute",
            "negative",
            "normal chest",
        ]

    lookup = {
        "glioma": ["glioma", "glial", "infiltrative", "edema", "ring", "mass effect"],
        "meningioma": ["meningioma", "extra-axial", "dural tail", "parasagittal", "sphenoid"],
        "pituitary": ["pituitary", "sellar", "suprasellar", "adenoma", "chiasm"],
        "no_tumor": ["normal", "no focal", "no pathology", "no evidence", "featureless"],
    }
    return lookup.get(prediction.lower(), [prediction.lower()])


def _keywords_for_location(disease_key: str, location: str) -> list[str]:
    lowered = location.lower()
    keywords: list[str] = [lowered]
    if disease_key == "tuberculosis":
        if "upper" in lowered:
            keywords.extend(["upper lobe", "apex", "apical"])
        if "lower" in lowered:
            keywords.extend(["lower lobe", "basilar", "base"])
        if "left" in lowered:
            keywords.append("left")
        if "right" in lowered:
            keywords.append("right")
        if "bilateral" in lowered:
            keywords.append("bilateral")
        if "central" in lowered or "mid" in lowered:
            keywords.extend(["mid", "central"])
        return keywords

    for token in [
        "left",
        "right",
        "frontal",
        "temporal",
        "parietal",
        "brainstem",
        "pons",
        "sellar",
        "suprasellar",
        "intrasellar",
        "pituitary",
        "cerebellopontine",
        "infratentorial",
        "parasagittal",
        "olfactory",
    ]:
        if token in lowered:
            keywords.append(token)
    if "sphenoid" in lowered:
        keywords.extend(["sphenoid", "temporal"])
    if "angle" in lowered:
        keywords.extend(["angle", "cerebellopontine"])
    return keywords


def _score_report(
    disease_key: str,
    report: dict[str, str],
    prediction: str,
    location: str,
) -> int:
    haystack = " ".join(
        [
            report["disease_type"],
            report["class_label"],
            report["location"],
            report["findings"],
            report["impression"],
        ]
    ).lower()

    score = 0
    for keyword in _keywords_for_prediction(disease_key, prediction):
        if keyword in haystack:
            score += 3
    for keyword in _keywords_for_location(disease_key, location):
        if keyword and keyword in haystack:
            score += 2
    return score


def select_reference_reports(
    disease_key: str,
    disease_type: str,
    prediction: str,
    location: str,
    limit: int = 4,
) -> list[SimpleReferenceReport]:
    reports = _load_reports(disease_key)
    filtered_reports = [
        report
        for report in reports
        if report["disease_type"].casefold() == disease_type.casefold()
        and report["class_label"].casefold() == prediction.casefold()
    ]
    candidate_reports = filtered_reports or reports
    scored = [
        SimpleReferenceReport(
            disease_type=report["disease_type"],
            class_label=report["class_label"],
            location=report["location"],
            findings=report["findings"],
            impression=report["impression"],
            score=_score_report(disease_key, report, prediction, location),
        )
        for report in candidate_reports
    ]
    ranked = sorted(scored, key=lambda item: item.score, reverse=True)
    filtered = [report for report in ranked if report.score > 0]
    return (filtered or ranked)[:limit]


def build_reference_context(reports: list[SimpleReferenceReport]) -> str:
    return "\n\n".join(
        [
            (
                f"Reference Report {index + 1}\n"
                f"Disease Type: {report.disease_type}\n"
                f"Class Label: {report.class_label}\n"
                f"Location: {report.location}\n"
                f"Findings: {report.findings}\n"
                f"Impression: {report.impression}"
            )
            for index, report in enumerate(reports)
        ]
    )


def generate_llm_api_explanation(
    disease_key: str,
    prediction: str,
    confidence_percent: float,
    location: str,
    context: str,
    model: str = DEFAULT_MODEL,
) -> str:
    if disease_key == "brain-tumor":
        system_prompt = (
            "You are a professional Neuroradiologist who explains the reasoning behind a brain MRI finding. "
            "The AI classification is for glioma, meningioma, no_tumor, or pituitary, and it is supplemented with Grad-CAM heatmaps."
        )
    else:
        system_prompt = (
            "You are a professional Radiologist who explains the reasoning of finding of chest X-ray. "
            "The AI classification is for Tuberculosis or Normal, and it is supplemented with Grad-CAM heatmaps."
        )

    prompt = f"""
SYSTEM: {system_prompt}
Use the provided Clinical Reference Reports to explain a new AI finding.

AI PREDICTION: {prediction}
CONFIDENCE: {confidence_percent:.2f}%
GRAD-CAM LOCATION: {location}

CLINICAL REFERENCE REPORTS FROM DATABASE:
{context}

TASK:
1. Summarize why the AI likely flagged the {location} based on the reference reports.
2. Explain the typical characteristics of {prediction} as described in the references.
3. Keep the tone professional and clinical. Do not provide a definitive diagnosis; use phrases like 'suggestive of' or 'consistent with'.
4. Keep your response limited to 100 words.
5. Remove any markdown formatting from your response.
""".strip()

    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")

    request = Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
    except URLError as error:
        raise RuntimeError(f"LLM API request failed: {error}") from error

    explanation = str(body.get("response", "")).strip()
    if not explanation:
        raise RuntimeError("LLM API returned an empty explanation.")
    return explanation
