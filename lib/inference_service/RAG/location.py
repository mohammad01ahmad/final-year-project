from __future__ import annotations

import re

import numpy as np


def extract_tb_location_from_heatmap(heatmap: np.ndarray) -> str:
    if heatmap is None or heatmap.size == 0:
        return "diffuse bilateral lung regions"

    working = np.nan_to_num(heatmap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    max_val = np.max(working)
    if max_val <= 0:
        return "diffuse bilateral lung regions"

    working = working / (max_val + 1e-8)
    working[working < 0.2] = 0

    height, width = working.shape
    mid_x = width // 2
    peak_y, peak_x = np.unravel_index(np.argmax(working), working.shape)
    left_max = float(np.max(working[:, :mid_x]))
    right_max = float(np.max(working[:, mid_x:]))
    is_bilateral = (left_max > 0.25) and (right_max > 0.25)

    rel_y = peak_y / height
    rel_x = peak_x / width
    if rel_y < 0.33:
        y_label = "apical (upper)"
    elif rel_y > 0.66:
        y_label = "basilar (lower)"
    else:
        y_label = "hilar (central)" if 0.35 < rel_x < 0.65 else "mid-zone"

    if is_bilateral:
        return f"bilateral {y_label} lung regions"

    side = "left" if peak_x < mid_x else "right"
    return f"{side} {y_label} lung region"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _brain_location_tokens(value: str) -> set[str]:
    normalized = _normalize_text(value)
    synonyms = {
        "frontal": {"frontal", "olfactory"},
        "temporal": {"temporal", "sphenoid"},
        "parietal": {"parietal", "parasagittal"},
        "brainstem": {"brainstem", "pons"},
        "pons": {"brainstem", "pons"},
        "sellar": {"sellar", "suprasellar", "pituitary"},
        "suprasellar": {"sellar", "suprasellar", "pituitary"},
        "intrasellar": {"sellar", "intrasellar", "pituitary"},
        "pituitary": {"sellar", "suprasellar", "intrasellar", "pituitary"},
        "cerebellopontine": {"cerebellopontine", "angle", "infratentorial"},
        "angle": {"cerebellopontine", "angle"},
        "infratentorial": {"infratentorial", "cerebellar", "posterior"},
        "left": {"left"},
        "right": {"right"},
        "n/a": {"n/a", "normal"},
        "normal": {"n/a", "normal"},
    }

    tokens: set[str] = set()
    for token in re.split(r"[^a-z0-9/]+", normalized):
        if not token:
            continue
        if token in synonyms:
            tokens.update(synonyms[token])
        else:
            tokens.add(token)

    if "parasagittal" in normalized:
        tokens.update({"parasagittal", "parietal", "right"})
    if "olfactory groove" in normalized:
        tokens.update({"olfactory", "frontal"})
    if "sphenoid wing" in normalized:
        tokens.update({"sphenoid", "temporal"})
    if "cerebellopontine angle" in normalized:
        tokens.update({"cerebellopontine", "angle", "infratentorial"})
    if "sellar region" in normalized:
        tokens.update({"sellar", "pituitary"})
    if "brainstem / pons" in normalized:
        tokens.update({"brainstem", "pons"})
    if normalized == "n/a":
        tokens.update({"n/a", "normal"})
    return tokens


def extract_brain_tumor_location_from_heatmap(heatmap: np.ndarray, prediction: str) -> str:
    if prediction == "no_tumor":
        return "N/A"

    if heatmap is None or heatmap.size == 0:
        return "Sellar Region"

    working = np.nan_to_num(heatmap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(np.max(working))
    if max_val <= 0:
        return "Sellar Region"

    normalized = working / (max_val + 1e-8)
    peak_y, peak_x = np.unravel_index(np.argmax(normalized), normalized.shape)
    height, width = normalized.shape
    rel_x = peak_x / max(width - 1, 1)
    rel_y = peak_y / max(height - 1, 1)

    center_dx = abs(rel_x - 0.5)
    center_dy = abs(rel_y - 0.2)
    if center_dx < 0.12 and center_dy < 0.12:
        if prediction == "pituitary":
            return "Sellar / Suprasellar"
        return "Sellar Region"

    if rel_y > 0.78:
        if center_dx < 0.18:
            return "Brainstem / Pons"
        return "Infratentorial"

    if rel_y > 0.62 and (rel_x < 0.18 or rel_x > 0.82):
        return "Cerebellopontine Angle"

    if rel_y < 0.2 and rel_x < 0.38:
        return "Olfactory Groove / Frontal"

    if rel_y < 0.28 and rel_x > 0.62:
        return "Parasagittal / Right Parietal"

    if rel_x < 0.42:
        return "Left Frontal Lobe" if rel_y < 0.45 else "Infratentorial"

    if rel_x > 0.6:
        return "Right Temporal Lobe" if rel_y < 0.46 else "Right Parietal Lobe"

    if prediction == "pituitary":
        return "Intrasellar"
    if prediction == "meningioma":
        return "Sphenoid Wing / Temporal"
    if prediction == "glioma":
        return "Left Frontal Lobe"
    return "Sellar Region"


def score_brain_reference_location(extracted_location: str, reference_location: str, prediction: str) -> int:
    extracted = _brain_location_tokens(extracted_location)
    reference = _brain_location_tokens(reference_location)
    if not reference:
        return 0

    score = len(extracted & reference) * 3
    extracted_normalized = _normalize_text(extracted_location)
    reference_normalized = _normalize_text(reference_location)

    if extracted_normalized == reference_normalized:
        score += 8
    if prediction == "pituitary" and {"sellar", "pituitary"} & reference:
        score += 4
    if prediction == "no_tumor" and ("n/a" in reference or "normal" in reference):
        score += 6
    if "left" in extracted and "left" in reference:
        score += 2
    if "right" in extracted and "right" in reference:
        score += 2
    return score
