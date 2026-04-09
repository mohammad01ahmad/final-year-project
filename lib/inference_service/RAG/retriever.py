from __future__ import annotations

from dataclasses import dataclass

from .kb import get_collection
from .location import score_brain_reference_location


@dataclass(frozen=True)
class RetrievedReport:
    id: str
    findings: str
    impression: str
    disease_type: str
    class_label: str
    location: str
    document: str
    distance: float | None


def _build_query(
    disease_type: str,
    prediction: str,
    confidence_percent: float,
    location: str,
) -> str:
    normalized_prediction = prediction.upper()
    if disease_type.casefold() == "tuberculosis":
        if normalized_prediction == "TUBERCULOSIS":
            diagnostic_hint = (
                "active tuberculosis findings: patchy opacities, cavitary lesions, "
                "hilar adenopathy, consolidation in upper lobes, pleural effusion, "
                "miliary patterns, acid-fast bacilli positive context."
            )
        else:
            diagnostic_hint = (
                "normal chest x-ray no radiographic evidence of active tuberculosis clear lungs "
                "no active cardiopulmonary disease"
            )
    else:
        diagnostic_hint = (
            "brain MRI lesion location, tumor morphology, mass effect, edema, "
            "extra-axial versus intra-axial appearance, and regional anatomy."
        )

    return (
        f"{disease_type} reference reports for prediction {prediction}. "
        f"Confidence {confidence_percent:.2f} percent. "
        f"Grad-CAM location {location}. "
        f"{diagnostic_hint}"
    )


def _format_context(matches: list[RetrievedReport]) -> str:
    return "\n\n".join(
        [
            (
                f"Reference Report {index + 1}\n"
                f"Disease Type: {match.disease_type}\n"
                f"Class Label: {match.class_label}\n"
                f"Location: {match.location}\n"
                f"Findings: {match.findings}\n"
                f"Impression: {match.impression}"
            )
            for index, match in enumerate(matches)
        ]
    )


def retrieve_rag_context(
    disease_key: str,
    disease_type: str,
    prediction: str,
    confidence_percent: float,
    location: str,
    top_k: int = 4,
) -> tuple[str, list[RetrievedReport]]:
    collection = get_collection(disease_key)
    query = _build_query(disease_type, prediction, confidence_percent, location)

    try:
        result = collection.query(
            query_texts=[query],
            n_results=max(top_k * 2, top_k),
            include=["documents", "metadatas", "distances"],
            where={
                "$and": [
                    {"disease_type": disease_type},
                    {"class_label": prediction},
                ]
            },
        )
    except Exception:
        result = collection.query(
            query_texts=[query],
            n_results=max(top_k * 2, top_k),
            include=["documents", "metadatas", "distances"],
        )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]

    matches: list[RetrievedReport] = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        identifier = ids[index] if index < len(ids) else f"{disease_key}-report-{index:05d}"
        matches.append(
            RetrievedReport(
                id=identifier,
                findings=str(metadata.get("findings", "")),
                impression=str(metadata.get("impression", "")),
                disease_type=str(metadata.get("disease_type", "")),
                class_label=str(metadata.get("class_label", "")),
                location=str(metadata.get("location", "")),
                document=str(document),
                distance=float(distance) if distance is not None else None,
            )
        )

    if disease_key == "brain-tumor":
        matches = sorted(
            matches,
            key=lambda match: (
                -score_brain_reference_location(location, match.location, prediction),
                match.distance if match.distance is not None else 9999.0,
            ),
        )

    matches = matches[:top_k]
    if not matches:
        return "No reference reports found for this prediction.", []

    return _format_context(matches), matches


def retrieve_tb_context(
    disease_type: str,
    prediction: str,
    confidence_percent: float,
    location: str,
    top_k: int = 4,
) -> tuple[str, list[RetrievedReport]]:
    return retrieve_rag_context(
        disease_key="tuberculosis",
        disease_type=disease_type,
        prediction=prediction,
        confidence_percent=confidence_percent,
        location=location,
        top_k=top_k,
    )


def retrieve_brain_tumor_context(
    disease_type: str,
    prediction: str,
    confidence_percent: float,
    location: str,
    top_k: int = 4,
) -> tuple[str, list[RetrievedReport]]:
    return retrieve_rag_context(
        disease_key="brain-tumor",
        disease_type=disease_type,
        prediction=prediction,
        confidence_percent=confidence_percent,
        location=location,
        top_k=top_k,
    )
