import Link from "next/link";

// Docs header: the BLAST wordmark, top-left, and nothing else. No other nav
// items, no login/signup/talk-to-us. Same fixed-bar + in-flow-spacer
// treatment as arker.ai's own nav (cloudflare/src/components/landing/nav-client.tsx),
// stripped down to just the logo per the brief.
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
