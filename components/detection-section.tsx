"use client";

import html2canvas from "html2canvas";
import { jsPDF } from "jspdf";
import { useEffect, useMemo, useRef, useState, useTransition } from "react";
import { DetectionConfig, DetectionResponse } from "@/lib/types/types";
import { RiUploadCloud2Fill } from "react-icons/ri";
import { RxReset } from "react-icons/rx";
import { MdZoomIn } from "react-icons/md";
import { FaArrowDown } from "react-icons/fa";
import { IoIosMedical } from "react-icons/io";
import { FiDownload } from "react-icons/fi";

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
  const reportRef = useRef<HTMLDivElement | null>(null);


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
    config.key === "alzheimers" || config.key === "tuberculosis" || config.key === "brain-tumor" || config.key === "chest-diseases";
  const supportsLlmExperiment =
    config.key === "alzheimers" || config.key === "tuberculosis" || config.key === "brain-tumor" || config.key === "chest-diseases";

  async function handleDownloadReport() {
    if (!response || !reportRef.current) return;

    try {
      const canvas = await html2canvas(reportRef.current, {
        scale: 2,
        // Force a plain white background so oklch background vars never reach the parser
        backgroundColor: "#ffffff",
        useCORS: true,
        logging: false,
        // Skip any element that could still carry a live theme variable
        ignoreElements: (el) =>
          el.classList.contains("ignore-canvas"),
        onclone: (_doc, element) => {
          // Make the hidden div visible to html2canvas during capture
          element.style.position = "relative";
          element.style.left = "0";
          element.style.opacity = "1";
        },
      });

      const imageData = canvas.toDataURL("image/png");

      const pdf = new jsPDF({
        orientation: "portrait",
        unit: "mm",
        format: "a4",
      });

      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const imgWidth = pageWidth;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;

      // FIX: correct multi-page offset math
      let heightLeft = imgHeight;
      let pageNumber = 0;

      pdf.addImage(imageData, "PNG", 0, 0, imgWidth, imgHeight);
      heightLeft -= pageHeight;

      while (heightLeft > 0) {
        pageNumber += 1;
        pdf.addPage();
        // Shift the image up by one full page each time
        pdf.addImage(imageData, "PNG", 0, -(pageHeight * pageNumber), imgWidth, imgHeight);
        heightLeft -= pageHeight;
      }

      const blob = pdf.output("blob");
      const blobUrl = URL.createObjectURL(blob);

      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = `${config.key}-diagnosis-report.pdf`;
      link.click();

      // FIX: always revoke, don't rely on beforeunload
      setTimeout(() => URL.revokeObjectURL(blobUrl), 10_000);
    } catch (err) {
      console.error("Report generation failed:", err);
    }
  }

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
          <p className="text-primary font-bold animate-pulse">Analyzing image via Medical Inference API...</p>
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
            <div className="md:col-span-1 bg-error-container/40 bg-red-100 p-6 rounded-xl border-l-4 border-error flex flex-col justify-between">
              <div>
                <h4 className="text-lg font-bold uppercase text-on-error-container mb-4 text-error">Predicted Class</h4>
                <div className="flex items-center gap-3">
                  <div className="bg-error/10 p-1.5 rounded-full">
                    <IoIosMedical className="text-4xl text-error" />
                  </div>
                  <div>
                    <span className="block font-headline text-2xl font-extrabold text-red-800 leading-tight">Diagnosis</span>
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
              <div className="flex items-center justify-between">
                <h4 className="font-bold uppercase tracking-widest text-on-surface-variant mb-4">
                  Class Probabilities
                </h4>
                <FaArrowDown className="text-on-surface-variant mb-4 mr-2 text-md" />
              </div>
              <div className="text-on-surface leading-relaxed text-sm md:text-base space-y-4 font-body pr-2">
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
                <p className="text-sm italic text-black">The AI model has highlighted the most salient regions in the generated heatmap overlay. Clinical correlation with the patient's reported symptoms is highly recommended.</p>
              </div>
            </div>
          </div>

          {supportsClinicalExplanation ? (
            <div className="mb-10 flex flex-col gap-6">
              <div className="w-full rounded-xl bg-surface-container-low p-6 shadow-[0_4px_20px_rgba(0,0,0,0.02)]">
                <div className="mb-4 flex items-center justify-between gap-4">
                  <h4 className="font-bold uppercase tracking-widest text-on-surface-variant">
                    RAG Extraction
                  </h4>
                  {response.ragSummary ? (
                    <div className="text-right text-xs text-on-surface-variant">
                      <p>Retrieved references: {response.ragSummary.retrievedCount}</p>
                    </div>
                  ) : null}
                </div>
                <div className="rounded-lg text-sm leading-7 text-on-surface">
                  {response.explanation ? (
                    <div className="space-y-4">
                      <p className="">{response.explanation}</p>
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

              {supportsLlmExperiment ? (
                <div className="w-full rounded-xl bg-[#004A7C] p-6 shadow-[0_4px_20px_rgba(0,0,0,0.02)]">
                  <div className="mb-4 flex items-center justify-between gap-4">
                    <h4 className="pl-2 pt-2 text-2xl font-bold uppercase tracking-widest text-white animate-pulse">
                      FINAL DIAGNOSIS
                    </h4>
                    <div className="flex items-center gap-4">
                      {response.llmApiSummary ? (
                        <div className="text-right text-xs text-sky-100">
                          <p>Selected reports: {response.llmApiSummary.selectedCount}</p>
                        </div>
                      ) : null}
                    </div>
                  </div>
                  <div className="rounded-lg pt-2 pb-2 pl-2 text-lg leading-7 text-[#D0E4FF]">
                    {response.llmApiExplanation ? (
                      <p>{response.llmApiExplanation}</p>
                    ) : (
                      <p className="italic text-sky-100">
                        No LLM API experiment output is available for this result right now.
                      </p>
                    )}
                  </div>
                </div>
              ) : null}
            </div>
          ) : null}

          <div className="pointer-events-none fixed left-[-10000px] top-0 z-[-1] opacity-0">
            <div
              ref={reportRef}
              style={{ fontFamily: "sans-serif" }}
              className="w-[794px] bg-white px-10 py-10"
            >
              {/* Header */}
              <div className="mb-8 border-b border-[#e2e8f0] pb-6">
                <p className="text-3xl font-extrabold uppercase tracking-[0.18em] text-[#004A7C]">
                  Medical Diagnosis
                </p>
                <p className="mt-3 text-xl font-semibold text-[#374151]">
                  {config.title.replace(" Detection", "")}
                </p>
              </div>

              {/* Images */}
              <div className="mb-8 grid grid-cols-2 gap-6">
                <div className="overflow-hidden rounded-2xl border border-[#e2e8f0] bg-[#f8fafc]">
                  <div className="border-b border-[#e2e8f0] px-4 py-3 text-sm font-bold uppercase tracking-[0.12em] text-[#475569]">
                    Original Radiograph
                  </div>
                  <div className="p-4">
                    {previewUrl ? (
                      <img
                        className="h-[250px] w-full rounded-xl object-cover"
                        src={previewUrl}
                        alt="Original uploaded image"
                        crossOrigin="anonymous"
                      />
                    ) : (
                      <div className="flex h-[250px] items-center justify-center rounded-xl bg-[#e2e8f0] text-sm text-[#64748b]">
                        No image available
                      </div>
                    )}
                  </div>
                </div>

                <div className="overflow-hidden rounded-2xl border border-[#e2e8f0] bg-[#f8fafc]">
                  <div className="border-b border-[#e2e8f0] px-4 py-3 text-sm font-bold uppercase tracking-[0.12em] text-[#475569]">
                    Grad-CAM Heatmap
                  </div>
                  <div className="p-4">
                    <img
                      className="h-[250px] w-full rounded-xl object-cover"
                      src={response.gradcamImage}
                      alt="Grad-CAM heatmap"
                      crossOrigin="anonymous"
                    />
                  </div>
                </div>
              </div>

              {/* Prediction */}
              <div className="mb-6 rounded-2xl border border-[#fecaca] bg-[#fef2f2] p-6">
                <div className="mb-3 text-sm font-bold uppercase tracking-[0.12em] text-[#dc2626]">
                  Predicted Class
                </div>
                <div className="text-3xl font-extrabold text-[#991b1b]">
                  {response.prediction}
                </div>
                <div className="mt-3 text-sm font-semibold text-[#7f1d1d]">
                  Confidence Score:{" "}
                  {sortedProbabilities.length > 0
                    ? formatPercent(sortedProbabilities[0].score)
                    : "0%"}
                </div>
              </div>

              {/* Probabilities */}
              <div className="mb-6 rounded-2xl border border-[#e2e8f0] bg-[#f8fafc] p-6">
                <div className="mb-4 text-sm font-bold uppercase tracking-[0.12em] text-[#475569]">
                  Class Probabilities
                </div>
                <div className="space-y-3">
                  {sortedProbabilities.map((prob) => (
                    <div key={`report-${prob.label}`}>
                      <div className="mb-1 flex items-center justify-between text-sm font-semibold text-[#1e293b]">
                        <span>{prob.label}</span>
                        <span>{formatPercent(prob.score)}</span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-[#e2e8f0]">
                        <div
                          className="h-full bg-[#004A7C]"
                          style={{ width: `${prob.score * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
                {response.ragSummary?.location ? (
                  <p className="mt-4 text-sm font-semibold text-[#374151]">
                    Grad-CAM location:{" "}
                    <span className="font-bold underline">
                      {response.ragSummary.location}
                    </span>
                  </p>
                ) : null}
              </div>

              {/* RAG */}
              <div className="mb-6 rounded-2xl border border-[#e2e8f0] bg-white p-6">
                <div className="mb-4 text-sm font-bold uppercase tracking-[0.12em] text-[#475569]">
                  RAG Extraction
                </div>
                <p className="whitespace-pre-wrap text-[15px] leading-7 text-[#1e293b]">
                  {response.explanation ??
                    "No clinical explanation is available for this result right now."}
                </p>
              </div>

              {/* Final diagnosis */}
              <div className="rounded-2xl bg-[#004A7C] p-6 text-[#D0E4FF]">
                <div className="mb-4 text-sm font-bold uppercase tracking-[0.12em] text-white">
                  Final Diagnosis
                </div>
                <p className="whitespace-pre-wrap text-[15px] leading-7">
                  {response.llmApiExplanation ??
                    "No final diagnosis summary is available for this result right now."}
                </p>
              </div>
            </div>
          </div>
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

      {/* Place this anywhere in the response view, e.g. at the top of the results section */}
      {response && (
        <div className="flex justify-end">
          <button
            type="button"
            onClick={handleDownloadReport}
            className="inline-flex items-center gap-2 rounded-full bg-[#004A7C] px-5 py-2.5 text-sm font-semibold text-white hover:bg-[#003a61] transition-colors"
          >
            <FiDownload className="text-base" />
            Download Report
          </button>
        </div>
      )}
    </>
  );
}
