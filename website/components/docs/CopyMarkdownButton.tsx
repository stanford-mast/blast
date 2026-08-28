"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

// Ports arker.ai/docs's CopyPageMenu (cloudflare/src/components/docs/copy-page-menu.tsx):
// same split button + chevron dropdown shape (bg-surface, hover:bg-surface-hover,
// duration-[50ms], full width, dropdown-panel-enter/exit), without
// lucide-react/sonner as dependencies (simple inline SVGs and an inline
// "Copied" state instead of a toast) and without the "Copy all pages" /
// "Open .md" / "Copy .md link" rows: those need a per-page raw-markdown
// source and a `.md` route BLAST's hand-written pages don't have, and a
// dropdown row that silently does the wrong thing is worse than one fewer
// row. The two rows here (copy this page, copy this page's link) are both
// real.
function CopyIcon() {
  return (
    <svg viewBox="0 0 24 24" className="size-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="0" ry="0" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg viewBox="0 0 24 24" className="size-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className="size-3.5 text-ink-tertiary transition-transform duration-150"
      style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

function LinkIcon() {
  return (
    <svg viewBox="0 0 24 24" className="size-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
    </svg>
  );
}

// Console-style copy glyph: Copy and Check cross-fading, same as the
// reference's CopyGlyph.
function CopyGlyph({ copied }: { copied: boolean }) {
  return (
    <span className="relative inline-flex size-3.5 shrink-0 items-center justify-center">
      <span className={"absolute transition-opacity duration-100 " + (copied ? "opacity-0" : "opacity-100")}>
        <CopyIcon />
      </span>
      <span className={"absolute transition-opacity duration-100 " + (copied ? "opacity-100" : "opacity-0")}>
        <CheckIcon />
      </span>
    </span>
  );
}

type RowDef = { icon: ReactNode; label: ReactNode; onClick: () => void };

export function CopyMarkdownButton({ articleSelector = ".doc-content" }: { articleSelector?: string }) {
  const [open, setOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [mainCopied, setMainCopied] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  useEffect(() => {
    if (open) {
      setMounted(true);
      return;
    }
    if (!mounted) return;
    const t = window.setTimeout(() => setMounted(false), 120);
    return () => window.clearTimeout(t);
  }, [open, mounted]);

  const copyPage = async () => {
    const article = document.querySelector<HTMLElement>(articleSelector);
    if (article) await navigator.clipboard.writeText(article.innerText);
    setMainCopied(true);
    window.setTimeout(() => setMainCopied(false), 1200);
  };
  const copyLink = async () => {
    await navigator.clipboard.writeText(window.location.href);
    setOpen(false);
  };

  const rowDefs: RowDef[] = [
    { icon: <CopyIcon />, label: "Copy this page", onClick: copyPage },
    { icon: <LinkIcon />, label: "Copy page link", onClick: copyLink },
  ];

  return (
    <div ref={ref} className="relative block w-full text-sm">
      <div className="flex w-full">
        <button
          type="button"
          onClick={copyPage}
          className="flex flex-1 items-center gap-2 whitespace-nowrap bg-surface px-3 py-1.5 text-ink transition-colors duration-50 hover:bg-surface-hover"
        >
          <CopyGlyph copied={mainCopied} />
          {mainCopied ? "Copied" : "Copy Markdown"}
        </button>
        <button
          type="button"
          aria-label="More copy options"
          onClick={() => setOpen((v) => !v)}
          className={
            "inline-flex items-center border-l border-black/30 px-2 py-1.5 text-ink transition-colors duration-50 " +
            (open ? "bg-surface-hover" : "bg-surface hover:bg-surface-hover")
          }
        >
          <ChevronIcon open={open} />
        </button>
      </div>

      {mounted && (
        <div className="absolute right-0 top-full z-10 w-full">
          <div
            className={
              "bg-surface-hover transition-[opacity,transform] duration-100 " +
              (open ? "opacity-100" : "pointer-events-none opacity-0")
            }
          >
            <div className="flex flex-col">
              {rowDefs.map((r, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={r.onClick}
                  className="flex w-full items-center gap-2 whitespace-nowrap px-3 py-2 text-left text-ink-secondary transition-colors duration-50 hover:bg-page hover:text-ink"
                >
                  {r.icon}
                  <span className="flex-1">{r.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
