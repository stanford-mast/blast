import type { ReactNode } from "react";

// Lightly syntax-highlight a type annotation (e.g. `String | null`,
// `"idle" | "running"`) so it reads like code. Same scheme as arker.ai/docs's
// Fields (cloudflare/src/components/docs/fields.tsx), minus the parts of its
// tokenizer this smaller field set never exercises.
function renderType(type: string) {
  const parts = type.split(/("[^"]*"|\s*\|\s*|,\s+)/g).filter((s) => s !== "");
  return parts.map((tok, i) => {
    let cls = "text-[#9bbcec]"; // type name
    if (/^"/.test(tok)) cls = "text-[#9ece6a]"; // string literal
    else if (/^(\s*\|\s*|,\s+)$/.test(tok) || tok.trim() === "null")
      cls = "text-ink-tertiary"; // separators / null
    return (
      <span key={i} className={cls}>
        {tok}
      </span>
    );
  });
}

/**
 * Parameter reference: each field is a row with its name/type/required flag
 * in a left rail and a description on the right.
 */
export function Fields({ children }: { children: ReactNode }) {
  return (
    <div className="my-5 overflow-hidden border border-white/[0.08]">
      {children}
    </div>
  );
}

export function Field({
  name,
  type,
  required,
  children,
}: {
  name: string;
  /** A type annotation: `String`, `Int`, `Bool`, `String | null`, ... */
  type?: string;
  /** `true` -> "Required"; a string is shown verbatim (e.g. "Required, one of"). */
  required?: boolean | string;
  children?: ReactNode;
}) {
  const isRequired = required === true || typeof required === "string";
  const flag = isRequired
    ? typeof required === "string"
      ? required
      : "Required"
    : null;

  return (
    <div className="grid gap-x-8 gap-y-2 border-b border-white/[0.08] p-5 last:border-b-0 sm:grid-cols-[minmax(0,13rem)_1fr]">
      <div className="space-y-1.5">
        <div className="font-mono text-sm font-semibold text-ink">{name}</div>
        {type && (
          <div className="font-mono text-xs leading-relaxed">
            {renderType(type)}
          </div>
        )}
        {flag && (
          <div
            className={
              "text-[11px] font-medium " +
              (isRequired ? "text-brand/[0.72]" : "text-ink-tertiary")
            }
          >
            {flag}
          </div>
        )}
      </div>
      <div className="min-w-0 [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
        {children}
      </div>
    </div>
  );
}
