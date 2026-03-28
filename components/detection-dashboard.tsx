"use client";

import { useMemo, useState } from "react";
import { DetectionConfig, detections } from "@/lib/types/types";
import { DetectionSection } from "./detection-section";

function ShellIcon({
  children,
  active = false,
}: {
  children: string;
  active?: boolean;
}) {
  return (
    <span
      className={[
        "inline-flex h-9 w-9 items-center justify-center rounded-xl text-sm font-semibold",
        active
          ? "bg-primary text-on-primary shadow-md"
          : "bg-slate-100 text-slate-500",
      ].join(" ")}
    >
      {children}
    </span>
  );
}

export function DetectionDashboard() {
  /*
  const recentUploads = useMemo(
    () => [
      {
        patient: "Patient #8842-X",
        study: "Cranial MRI (T2 Weighted) • 42.1 MB",
        status: "Processing",
        tone: "bg-tertiary-container/30 text-on-tertiary-container",
        time: "2 mins ago",
      },
      {
        patient: "Patient #1029-C",
        study: "Chest CT w/ Contrast • 128 MB",
        status: "Analysis Ready",
        tone: "bg-secondary-container/30 text-on-secondary-container",
        time: "1 hour ago",
      },
      {
        patient: "Patient #4419-M",
        study: "Hand/Wrist X-Ray • 8.4 MB",
        status: "Analysis Ready",
        tone: "bg-secondary-container/30 text-on-secondary-container",
        time: "3 hours ago",
      },
    ],
    [],
  );
  */

  const [selectedDetection, setSelectedDetection] = useState<DetectionConfig>(detections[0]);

  return (
    <div className="flex min-h-screen bg-surface text-on-surface">

      {/* Side bar */}
      <aside className="fixed left-0 top-0 hidden h-screen w-72 overflow-y-auto border-r border-slate-200/70 bg-slate-50/90 p-4 backdrop-blur md:flex md:flex-col">
        <div className="mb-8 px-3">
          <h2 className="font-headline text-lg font-extrabold text-sky-900">
            Aether Medical
          </h2>
          <p className="mt-1 text-[11px] uppercase tracking-[0.24em] text-slate-500">
            Clinical Imaging
          </p>
        </div>

        <div className="flex flex-1 flex-col gap-2">
          {detections.map((d) => {
            const isActive = selectedDetection.key === d.key;
            return (
              <button
                key={d.key}
                onClick={() => setSelectedDetection(d)}
                className={[
                  "flex w-full items-center gap-3 rounded-2xl p-3 transition-all",
                  isActive
                    ? "bg-white text-sky-900 shadow-sm"
                    : "text-slate-500 hover:bg-white/60"
                ].join(" ")}
              >
                <ShellIcon active={isActive}>
                  {d.title.split(' ').map(word => word[0]).join('').toUpperCase().slice(0, 2)}
                </ShellIcon>
                <div className="text-left">
                  <p className="text-sm font-semibold">{d.title}</p>
                  {isActive && <p className="text-[10px] text-slate-500">Active diagnostics workspace</p>}
                </div>
              </button>
            );
          })}
        </div>

        <div className="border-t border-slate-200 pt-4">
          <a className="flex items-center gap-3 rounded-2xl p-3 text-slate-500 hover:bg-white">
            <ShellIcon>⚙</ShellIcon>
            <span className="text-sm font-medium">Settings</span>
          </a>
          <a className="flex items-center gap-3 rounded-2xl p-3 text-slate-500 hover:bg-white">
            <ShellIcon>?</ShellIcon>
            <span className="text-sm font-medium">Support</span>
          </a>
        </div>
      </aside>

      <main className="flex min-h-screen flex-1 flex-col md:ml-72">
        {/* Header */}
        <header className="sticky top-0 z-40 border-b border-slate-200/50 bg-white/85 shadow-[0_8px_32px_rgba(25,28,29,0.06)] backdrop-blur-md">
          <div className="mx-auto flex w-full max-w-[1920px] items-center justify-between px-6 py-3">
            <div className="flex items-center gap-8">
              <span className="text-xl font-extrabold tracking-tighter text-sky-950 md:hidden">
                AM
              </span>
              <nav className="hidden items-center gap-6 lg:flex">
                <a className="font-headline text-sm font-medium tracking-tight text-slate-500 hover:text-sky-800">
                  Analysis History
                </a>
                <a className="border-b-2 border-sky-900 pb-1 font-headline text-sm font-bold tracking-tight text-sky-900">
                  Upload New
                </a>
              </nav>
            </div>
            <div className="flex items-center gap-4">
              <div className="relative hidden sm:block">
                <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-sm text-slate-400">
                  Search
                </span>
                <input
                  className="w-48 rounded-full border border-transparent bg-slate-100 py-2 pl-18 pr-4 text-xs outline-none focus:border-sky-300 focus:ring-2 focus:ring-sky-500/20 lg:w-64"
                  placeholder="Search patient ID..."
                  type="text"
                />
              </div>
              <button className="inline-flex h-11 w-11 items-center justify-center rounded-full bg-slate-100 text-sm font-semibold text-sky-900 hover:bg-slate-200">
                AC
              </button>
            </div>
          </div>
        </header>

        {/* Upload Section */}
        <section className="mx-auto w-full max-w-7xl flex-1 px-6 py-8 lg:px-10">
          <DetectionSection config={selectedDetection} />

          {/* Patient History Section */}
          {/* <div className="panel-shadow overflow-hidden rounded-[1.75rem] border border-outline-variant/20 bg-surface-container-lowest">
            <div className="flex items-center justify-between border-b border-surface-container px-8 py-6">
              <h2 className="font-headline text-xl font-bold">Recent Uploads</h2>
              <button className="text-sm font-semibold text-primary hover:underline">
                View All History
              </button>
            </div>
            <div className="divide-y divide-surface-container">
              {recentUploads.map((upload) => (
                <div
                  key={upload.patient}
                  className="flex items-center justify-between gap-4 px-8 py-5 hover:bg-surface-bright"
                >
                  <div className="flex items-center gap-4">
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-surface-container-high text-xs font-bold text-slate-500">
                      IMG
                    </div>
                    <div>
                      <p className="text-sm font-bold">{upload.patient}</p>
                      <p className="text-xs text-on-surface-variant">{upload.study}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-6">
                    <span
                      className={`rounded-full px-3 py-1 text-[11px] font-bold ${upload.tone}`}
                    >
                      {upload.status}
                    </span>
                    <span className="hidden text-xs text-on-surface-variant md:block">
                      {upload.time}
                    </span>
                    <button className="text-outline hover:text-primary">•••</button>
                  </div>
                </div>
              ))}
            </div>
          </div> */}
        </section>

        <footer className="mt-auto w-full border-t border-slate-200/50 bg-slate-50 py-8">
          <div className="flex flex-col items-center justify-between gap-4 px-12 md:flex-row">
            <div className="flex flex-col items-center md:items-start">
              <span className="font-headline font-bold text-slate-900">
                Aether Medical Imaging
              </span>
              <span className="mt-1 text-[11px] uppercase tracking-[0.24em] text-slate-500">
                © 2024 Aether Medical Imaging. Clinical Grade Precision.
              </span>
            </div>
            <div className="flex flex-wrap justify-center gap-6">
              <a className="text-[11px] uppercase tracking-[0.24em] text-slate-400 hover:text-sky-600">
                HIPAA Compliance
              </a>
              <a className="text-[11px] uppercase tracking-[0.24em] text-slate-400 hover:text-sky-600">
                GDPR Policy
              </a>
              <a className="text-[11px] uppercase tracking-[0.24em] text-slate-400 hover:text-sky-600">
                Security Architecture
              </a>
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}
