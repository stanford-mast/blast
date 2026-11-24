import Link from "next/link";

// Docs header: the BLAST wordmark, top-left, and nothing else. No other nav
// items, no login/signup/talk-to-us. A fixed bar plus an in-flow spacer
// holding just the logo.
export function DocsHeader() {
  return (
    <>
      <div aria-hidden style={{ height: 60 }} />
      <header className="fixed inset-x-0 top-0 z-40 flex h-[60px] items-center bg-page px-6">
        <Link href="/" className="flex items-center gap-2.5">
          <img src="/blast_icon_only.svg" alt="" className="h-6 w-6 shrink-0" />
          <span className="font-display text-xl font-medium tracking-widest text-ink">
            BLAST
          </span>
        </Link>
      </header>
    </>
  );
}
