import Head from "next/head";
import type { ReactNode } from "react";
import { DocsHeader } from "./DocsHeader";
import { DocsNav } from "./DocsNav";
import { OnThisPage } from "./OnThisPage";
import { CopyMarkdownButton } from "./CopyMarkdownButton";

export type TocItem = { id: string; text: string };

// Shared docs shell: header (logo only) + left sidebar + centered content
// column + a right rail. Mirrors the three-column layout of arker.ai/docs's
// DocsDocument (cloudflare/src/app/docs/docs-document.tsx): the rail is Copy
// Markdown on top, the outline below, in ONE sticky flex column that travels
// with the content, not a separate/narrower column, which is what was making
// the page read as narrower than the reference. The rail always renders
// (every docs page gets Copy Markdown, even ones without an outline, e.g.
// the Introduction and the Guides); the outline itself only renders when the
// page supplies a toc. No visible "On this page" heading, the reference
// only carries that string as an aria-label, never rendered text
// (on-this-page.tsx:188).
export function DocsLayout({
  children,
  toc,
}: {
  children: ReactNode;
  toc?: TocItem[];
}) {
  return (
    <div className="flex min-h-screen flex-col bg-page text-ink">
      <Head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="icon" href="/blast_icon_only.svg" type="image/svg+xml" />
      </Head>
      <DocsHeader />
      <div className="flex w-full flex-1">
        <aside className="hidden w-64 shrink-0 border-r border-white/[0.06] lg:block">
          <div className="sticky top-[60px] h-[calc(100svh-60px)] overflow-y-auto px-5 py-8">
            <DocsNav />
          </div>
        </aside>

        <div className="flex min-w-0 flex-1 justify-center px-6 py-10 sm:px-10">
          <div className="flex w-full max-w-[64rem] gap-10 xl:gap-14">
            <main className="min-w-0 flex-1">
              <article className="doc-content">{children}</article>
            </main>

            <aside className="hidden w-64 shrink-0 xl:block">
              <div className="sticky top-[100px] flex flex-col gap-10">
                <CopyMarkdownButton />
                {toc && toc.length > 0 && <OnThisPage items={toc} />}
              </div>
            </aside>
          </div>
        </div>
      </div>
    </div>
  );
}
