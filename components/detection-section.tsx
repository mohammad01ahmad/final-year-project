"use client";

import { useEffect, useMemo, useState, useTransition } from "react";
import { DetectionConfig, detections, Probability, DetectionResponse, accentMap } from "@/lib/types/types";
import { RiUploadCloud2Fill } from "react-icons/ri";
import { RxReset } from "react-icons/rx";
import { MdZoomIn } from "react-icons/md";

const apiBase =
  process.env.NEXT_PUBLIC_INFERENCE_API_URL?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000";

function formatPercent(score: number) {
  return `${(score * 100).toFixed(2)}%`;
}

export function DetectionSection({ config }: { config: DetectionConfig }) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<DetectionResponse | null>(null);
  const [referencesExpanded, setReferencesExpanded] = useState(false);
  const [isPending, startTransition] = useTransition();


  // Preview the uploaded image
  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  // Sort probabilities in descending order
  const sortedProbabilities = useMemo(() => {
    return [...(response?.probabilities ?? [])].sort((a, b) => b.score - a.score);
  }, [response]);

  // Handle file selection
  function onFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0] ?? null;
    if (nextFile) {
      setSelectedFile(nextFile);
      setResponse(null);
      setReferencesExpanded(false);
      setError(null);
      analyzeImage(nextFile);
    }
  }

  // Analyze the uploaded image
  function analyzeImage(fileToAnalyze: File) {
    if (!fileToAnalyze) {
      setError("Please select an image before starting the analysis.");
      return;
    }

    startTransition(async () => {
      try {
        setError(null);
        setResponse(null);

        const formData = new FormData();
        formData.append("file", fileToAnalyze);

        const result = await fetch(`${apiBase}${config.endpoint}`, {
          method: "POST",
          body: formData,
        });

        if (!result.ok) {
          const detail = await result.text();
          throw new Error(detail || "Inference service returned an error.");
        }

        const payload = (await result.json()) as DetectionResponse;
        setResponse(payload);
      } catch (caughtError) {
        const message =
          caughtError instanceof Error
            ? caughtError.message
            : "Unable to complete the analysis request.";
        setError(message);
      }
    });
  }

  // Reset the analysis
  function handleReset() {
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
    setResponse(null);
    setReferencesExpanded(false);
  }

  const supportsClinicalExplanation =
    config.key === "tuberculosis" || config.key === "brain-tumor";

  return (
    <>
      <div className="mb-12 flex justify-between">
        <div>
          <h1 className="text-display-md mb-2 max-w-4xl text-4xl font-extrabold text-on-surface lg:text-5xl">
            {config.title}
          </h1>
          <div className="flex flex-col gap-2">
            <p className="max-w-3xl text-lg leading-relaxed text-on-surface-variant">
              {config.subtitle} — {config.acceptedImageHint}
            </p>
            <div className="flex flex-wrap gap-2">
              {config.classLabels.map((label) => (
                <span
                  key={label}
                  className="rounded-full bg-slate-100 px-3 py-1 text-md font-bold uppercase tracking-wider text-slate-600 border border-slate-200"
                >
                  {label}
                </span>
              ))}
            </div>
          </div>
        </div>
        <div className="flex items-end">
          <button onClick={() => handleReset()} className="cursor-pointer flex items-center gap-2 bg-primary text-on-primary text-lg rounded-full px-4 py-2"><RxReset /> Reset</button>
        </div>
      </div>

      {/* Results View */}
      {isPending ? (
        <div className="flex justify-center items-center py-20 flex-col gap-4">
          <div className="h-12 w-12 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
          <p className="text-primary font-bold animate-pulse">Analyzing image via Aether Medical Inference API...</p>
        </div>
      ) : response ? (
        <>
          {/* <!-- Analysis Split View --> */}
          <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* <!-- Original Image --> */}
            <div className="bg-surface-container-lowest p-4 rounded-xl shadow-[0_4px_20px_rgba(0,0,0,0.02)] flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-md font-bold uppercase tracking-widest text-on-surface-variant">Original Radiograph</h3>
              </div>
              <div className="relative aspect-square bg-black rounded-lg overflow-hidden group">
                {previewUrl && <img className="w-full h-full object-cover opacity-90 transition-opacity hover:opacity-100" src={previewUrl} alt="Original uploaded image" />}
                <div className="absolute items-end bottom-4 right-4 flex gap-2">
                  <button className="bg-black/50 backdrop-blur-md p-2 rounded-full text-white hover:bg-black/70 transition-all">
                    <span className="text-xl"><MdZoomIn /></span>
                  </button>
                </div>
              </div>
            </div>
            {/* <!-- Saliency Map overlay --> */}
            <div className="bg-surface-container-lowest p-4 rounded-xl shadow-[0_4px_20px_rgba(0,0,0,0.02)] flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-md font-bold uppercase tracking-widest text-on-surface-variant">grad-CAM Heatmap</h3>
                <div className="flex items-center gap-2">
                  <div className="h-3 w-16 bg-gradient-to-r from-blue-500 via-yellow-400 to-red-500 rounded-full"></div>
                  <span className="text-md text-on-surface-variant">Interest Scale</span>
                </div>
              </div>
              <div className="relative aspect-square bg-black rounded-lg overflow-hidden">
                <img className="w-full h-full object-cover" src={response.gradcamImage} alt="AI Saliency Map Overlay" />
                <div className="absolute inset-0 bg-primary/10 pointer-events-none"></div>
              </div>
            </div>
          </section>
          {/* <!-- Diagnosis & Commentary Bento Grid --> */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
            {/* <!-- Findings Panel --> */}
            <div className="md:col-span-1 bg-error-container/40 p-6 rounded-xl border-l-4 border-error flex flex-col justify-between">
              <div>
                <h4 className="text-lg font-bold uppercase tracking-widest text-on-error-container mb-4">Predicted Class</h4>
                <div className="flex items-start gap-3">
                  <div>
                    <span className="block font-headline text-2xl font-extrabold text-on-error-container leading-tight">Diagnosis</span>
                    <span className="text-error font-bold text-lg">({response.prediction})</span>
                  </div>
                </div>
              </div>
              <div className="mt-6">
                <div className="flex justify-between text-[18px] font-bold text-on-error-container/70 mb-1">
                  <span>Confidence Score</span>
                  <span>{sortedProbabilities.length > 0 ? formatPercent(sortedProbabilities[0].score) : "0%"}</span>
                </div>
                <div className="w-full bg-error-container h-1.5 rounded-full overflow-hidden">
                  <div className="bg-error h-full" style={{ width: sortedProbabilities.length > 0 ? `${sortedProbabilities[0].score * 100}%` : "0%" }}></div>
                </div>
              </div>
            </div>
            {/* <!-- Expert Commentary --> */}
            <div className="md:col-span-2 bg-surface-container-low p-6 rounded-xl flex flex-col">
              <h4 className="font-bold uppercase tracking-widest text-on-surface-variant mb-4 flex items-center gap-2">
                Class Probabilities
              </h4>
              <div className="text-on-surface leading-relaxed text-sm md:text-base space-y-4 font-body custom-scrollbar overflow-y-auto max-h-[200px] pr-2">
                <div className="flex flex-col gap-3">
                  {sortedProbabilities.map((prob) => (
                    <div key={prob.label} className="w-full">
                      <div className="flex justify-between text-md mb-1 font-semibold">
                        <span>{prob.label}</span>
                        <span>{formatPercent(prob.score)}</span>
                      </div>
                      <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
                        <div className="bg-primary h-full" style={{ width: `${prob.score * 100}%` }}></div>
                      </div>
                    </div>
                  ))}
                </div>
                {supportsClinicalExplanation && response.ragSummary?.location ? (
                  <p className="mt-4 text-lg font-semibold text-on-surface">
                    Grad-CAM location: <span className="font-bold underline">{response.ragSummary.location}</span>
                  </p>
                ) : null}
                <p className="mt-4 text-md italic text-black">The AI model has highlighted the most salient regions in the generated heatmap overlay. Clinical correlation with the patient's reported symptoms is highly recommended.</p>
              </div>
            </div>
          </div>

          {supportsClinicalExplanation ? (
            <div className="mb-10 grid grid-cols-1 gap-6 xl:grid-cols-2">
              <div className="rounded-xl bg-surface-container-low p-6 shadow-[0_4px_20px_rgba(0,0,0,0.02)]">
                <div className="mb-4 flex items-center justify-between gap-4">
                  <h4 className="font-bold uppercase tracking-widest text-on-surface-variant">
                    Clinical Explanation
                  </h4>
                  {response.ragSummary ? (
                    <div className="text-right text-xs text-on-surface-variant">
                      <p>Retrieved references: {response.ragSummary.retrievedCount}</p>
                    </div>
                  ) : null}
                </div>
                <div className="rounded-lg bg-white/80 p-5 text-sm leading-7 text-on-surface">
                  {response.explanation ? (
                    <div className="space-y-4">
                      <p>{response.explanation}</p>
                      {response.ragReferences ? (
                        <div className="rounded-lg border border-slate-200 bg-slate-50">
                          <button
                            type="button"
                            onClick={() => setReferencesExpanded((current) => !current)}
                            className="flex w-full items-center justify-between px-4 py-3 text-left"
                          >
                            <span className="text-xs font-bold uppercase tracking-widest text-on-surface-variant">
                              References
                            </span>
                            <span className="text-sm text-on-surface-variant">
                              {referencesExpanded ? "▲" : "▼"}
                            </span>
                          </button>
                          {referencesExpanded ? (
                            <div className="border-t border-slate-200 px-4 pb-4 pt-3">
                              <pre className="whitespace-pre-wrap font-body text-sm leading-6 text-on-surface">
                                {response.ragReferences}
                              </pre>
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <p className="italic text-on-surface-variant">
                      No clinical explanation is available for this result right now. The CNN prediction and Grad-CAM output are still valid.
                    </p>
                  )}
                </div>
              </div>

              {supportsClinicalExplanation ? (
                <div className="rounded-xl bg-surface-container-low p-6 shadow-[0_4px_20px_rgba(0,0,0,0.02)]">
                  <div className="mb-4 flex items-center justify-between gap-4">
                    <h4 className="font-bold uppercase tracking-widest text-on-surface-variant">
                      LLM API Experiment
                    </h4>
                    {response.llmApiSummary ? (
                      <div className="text-right text-xs text-on-surface-variant">
                        <p>Selected reports: {response.llmApiSummary.selectedCount}</p>
                      </div>
                    ) : null}
                  </div>
                  <div className="rounded-lg bg-white/80 p-5 text-sm leading-7 text-on-surface">
                    {response.llmApiExplanation ? (
                      <p>{response.llmApiExplanation}</p>
                    ) : (
                      <p className="italic text-on-surface-variant">
                        No LLM API experiment output is available for this result right now.
                      </p>
                    )}
                  </div>
                </div>
              ) : null}
            </div>
          ) : null}
        </>
      ) : (
        <>
          {error && (
            <div className="mb-6 pt-4 pb-4 bg-error-container  text-on-error-container rounded-lg font-medium">
              Error API Response: {error}
            </div>
          )}
          <label className={[
            "lg:col-span-8 bg-surface-container-lowest rounded-2xl p-10 border border-outline-variant/15 shadow-sm group cursor-pointer hover:bg-surface-bright transition-all flex flex-col items-center justify-center text-center space-y-6 min-h-[400px]",
            `hover:border-${config.accent}-400` // Dynamic hover border
          ].join(" ")}>
            <input type="file" className="hidden" accept="image/*" onChange={onFileChange} />
            <div className={`w-20 h-20 bg-${config.accent}-100 rounded-full flex items-center justify-center text-${config.accent}-600 mb-2`}>
              <RiUploadCloud2Fill className="text-5xl" />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-on-surface mb-2">Drag and drop imaging files</h3>
              <p className="text-on-surface-variant">or <span className="text-primary font-bold underline">browse files</span> from your workstation</p>
            </div>
          </label>
        </>
      )}

    </>
  );
}
