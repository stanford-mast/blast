import { Fragment } from "react";
import Link from "next/link";
import { endpointSlug } from "@/lib/docs/endpoint-slug";

// HTTP method -> badge tint. Same muted, dark-theme-friendly fills as
// arker.ai/docs's Endpoint (cloudflare/src/components/docs/endpoint.tsx) so
// the verb reads at a glance.
const METHOD_STYLES: Record<string, string> = {
  GET: "bg-[#13313f] text-[#7cc7e8]",
  POST: "bg-[#143126] text-[#6fd29b]",
  PUT: "bg-[#3a3115] text-[#e0c069]",
  PATCH: "bg-[#3a3115] text-[#e0c069]",
  DELETE: "bg-[#3a1a1a] text-[#e88a8a]",
};

// Split a path into static text + highlighted `{param}` segments.
function renderPath(path: string) {
  const parts = path.split(/(\{[^}]+\})/g);
  return parts.map((part, i) => {
    if (!part) return null;
    const isParam = /^\{[^}]+\}$/.test(part);
    return (
      <Fragment key={i}>
        {isParam ? <span className="text-brand">{part}</span> : part}
      </Fragment>
    );
  });
}

/**
 * API endpoint header: a method badge + path in a monospace surface, used in
 * place of a plain `# Title` on API reference pages. Renders as the page's h1
 * by default. Pass `href` to make the whole row a link (e.g. an operations
 * index linking out to each operation's own section).
 */
export function Endpoint({
  method,
  path,
  note,
  href,
  title,
  as: As = "h1",
}: {
  method: string;
  path: string;
  /** Short trailing label, e.g. "list VMs", when stacking endpoints. */
  note?: string;
  /** When set, the row links here (a doc page + optional `#section`). */
  href?: string;
  /** Human label for this operation (e.g. "Create session"), surfaced as the
      heading's title attribute; the route badge is still what's shown. */
  title?: string;
  /** `h1` for the page header, `h2` for a per-operation section header (gets
      an anchor id), `div` when stacking several endpoints under a heading. */
  as?: "h1" | "h2" | "div";
}) {
  const m = method.toUpperCase();
  const big = As === "h1" || As === "h2";
  const rowH = big ? "h-11" : "h-9";
  const routeFont = big ? "text-xl sm:text-2xl" : "text-base sm:text-lg";
  const badgeFont = big ? "text-2xl sm:text-3xl" : "text-lg sm:text-xl";

  // When `href` makes the row a link, every segment gets the same
  // group-hover feedback as arker.ai/docs's Endpoint (endpoint.tsx:70-105):
  // the badge brightens, the route surface lightens, and the trailing note
  // shifts to a brighter tone with a small arrow that slides into view.
  const body = (
    <>
      <span
        className={
          "inline-flex shrink-0 items-center justify-center px-2 font-mono font-bold leading-none tracking-wide " +
          (href ? "transition-[filter] duration-50 group-hover:brightness-125 " : "") +
          `${rowH} ${badgeFont} ` +
          (METHOD_STYLES[m] ?? METHOD_STYLES.POST)
        }
      >
        {m}
      </span>
      <span
        className={
          "inline-flex items-center bg-surface px-3 font-mono text-ink transition-colors duration-50 " +
          (href ? "group-hover:bg-surface-hover " : "") +
          `${rowH} ${routeFont}`
        }
      >
        {renderPath(path)}
      </span>
      {note && (
        <span
          className={
            "inline-flex items-center gap-1.5 text-sm text-ink-tertiary transition-colors duration-50 " +
            (href ? "group-hover:text-ink-secondary" : "")
          }
        >
          {note}
          {href && (
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
              className="size-3.5 shrink-0 opacity-0 transition-all duration-50 group-hover:translate-x-0.5 group-hover:opacity-100"
            >
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          )}
        </span>
      )}
    </>
  );

  const isHeading = As === "h2";
  const id = isHeading ? endpointSlug(method, path) : undefined;
  const rowClass =
    "flex flex-wrap items-center gap-3 " +
    (isHeading ? "mt-10 mb-3 scroll-mt-24" : "my-3");

  const content =
    href != null ? (
      <Link href={href} className="group flex flex-1 flex-wrap items-center gap-3 no-underline">
        {body}
      </Link>
    ) : (
      body
    );

  return (
    <As id={id} title={title} className={rowClass}>
      {content}
    </As>
  );
}
