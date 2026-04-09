export type DetectionConfig = {
    key: string;
    title: string;
    subtitle: string;
    endpoint: string;
    acceptedImageHint: string;
    classLabels: string[];
    accent: "sky" | "cyan" | "teal" | "amber";
};

export const detections: DetectionConfig[] = [
    {
        key: "alzheimers",
        title: "Alzheimers Detection",
        subtitle: "Brain MRI · 4-class dementia staging",
        endpoint: "/predict/alzheimers",
        acceptedImageHint: "MRI brain image in PNG, JPG, or JPEG format.",
        classLabels: [
            "Mild Demented",
            "Moderate Demented",
            "Non Demented",
            "Very MildDemented",
        ],
        accent: "sky",
    },
    {
        key: "brain-tumor",
        title: "Brain tumor Detection",
        subtitle: "MRI · 4-class tumor categorization",
        endpoint: "/predict/brain-tumor",
        acceptedImageHint: "MRI scan image in PNG, JPG, or JPEG format.",
        classLabels: ["glioma", "meningioma", "no_tumor", "pituitary"],
        accent: "cyan",
    },
    {
        key: "chest-diseases",
        title: "Chest diseases Detection",
        subtitle: "X-ray · COVID-19 / Normal / Non-COVID",
        endpoint: "/predict/chest-diseases",
        acceptedImageHint: "Chest X-ray image in PNG, JPG, or JPEG format.",
        classLabels: ["COVID-19", "Normal", "Non-COVID"],
        accent: "teal",
    },
    {
        key: "tuberculosis",
        title: "Tuberculosis Detection",
        subtitle: "X-ray · Binary tuberculosis screening",
        endpoint: "/predict/tuberculosis",
        acceptedImageHint: "Chest X-ray image in PNG, JPG, or JPEG format.",
        classLabels: ["NORMAL", "TUBERCULOSIS"],
        accent: "amber",
    },
];

export const accentMap: Record<DetectionConfig["accent"], string> = {
    sky: "from-sky-100/80 to-white text-sky-900 ring-sky-200",
    cyan: "from-cyan-100/80 to-white text-cyan-900 ring-cyan-200",
    teal: "from-teal-100/80 to-white text-teal-900 ring-teal-200",
    amber: "from-amber-100/80 to-white text-amber-900 ring-amber-200",
};

export type Probability = {
    label: string;
    score: number;
};

export type DetectionResponse = {
    prediction: string;
    probabilities: Probability[];
    gradcamImage: string;
    model: string;
    inputSize?: {
        width: number;
        height: number;
    };
    explanation?: string | null;
    ragSummary?: {
        location: string;
        retrievedCount: number;
    } | null;
    ragReferences?: string | null;
    llmApiExplanation?: string | null;
    llmApiSummary?: {
        location: string;
        selectedCount: number;
    } | null;
};
