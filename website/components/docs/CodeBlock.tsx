import { useState } from "react";

// Standalone doc code block: borderless figure, dark header bar with a
// language label + copy button, matching arker.ai/docs's code chrome
// (cloudflare/src/app/globals.css:940-1021, .doc-code-header /
// .doc-code-lang / .doc-code-copy / figure pre). `html` is real Shiki
// output (github-dark-dimmed, the same theme arker.ai/docs renders with),
// pre-highlighted at build time via lib/docs/highlight.ts + getStaticProps
// -- no client-side highlighter, no flash of unhighlighted code. `code` is
// the raw source, kept alongside purely for the copy button (copying
// should yield plain text, not the highlighted HTML). The nested [&_pre]
// overrides neutralize Shiki's own inline background/margin/padding, since
// this wrapper supplies all of that itself (same "strip Shiki's own
// background, force #181818" approach as arker's own
// `figure[data-rehype-pretty-code-figure] pre { background-color: #181818
// !important }").
export function CodeBlock({
  html,
  code,
  lang = "bash",
}: {
  html: string;
  code: string;
  lang?: string;
}) {
  const [copied, setCopied] = useState(false);

  const copy = () => {
    void navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };

  return (
    // No outer border: arker.ai's figure[data-rehype-pretty-code-figure]
    // (globals.css:940-946) has none; separation comes purely from the
    // header/body background split plus the header's own bottom border.
    // Margin is the figure's literal 1.25rem 0 (my-5, not my-6).
    <div className="my-5 overflow-hidden bg-[#181818]">
      <div className="flex items-center justify-between border-b border-border bg-[#131315] px-3.5 py-2">
        <span className="font-mono text-[11px] font-medium tracking-normal text-ink-tertiary">
          {lang}
        </span>
        <button
          type="button"
          aria-label={copied ? "Copied" : "Copy"}
          onClick={copy}
          className="inline-flex size-7 items-center justify-center text-ink-tertiary transition-colors duration-150 hover:bg-white/[0.06] hover:text-ink"
        >
          {copied ? (
            <svg viewBox="0 0 24 24" className="size-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" className="size-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="9" y="9" width="13" height="13" rx="0" ry="0" />
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
            </svg>
          )}
        </button>
      </div>
      <div
        className="overflow-x-auto px-4 py-[0.95rem] font-mono text-[13px] leading-[1.65] [tab-size:2] [&_pre]:!m-0 [&_pre]:!bg-transparent [&_pre]:!p-0 [&_code]:!bg-transparent"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}
