import { createHighlighter, type Highlighter } from "shiki";

// Same theme arker.ai/docs uses (cloudflare/src/lib/shiki-highlighter.ts):
// rehype-pretty-code renders with github-dark-dimmed. BLAST's docs only use
// bash and json code blocks, so that's the whole lang set.
const THEME = "github-dark-dimmed";
const LANGS = ["bash", "json"];

let highlighterPromise: Promise<Highlighter> | null = null;

function getHighlighter(): Promise<Highlighter> {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighter({ themes: [THEME], langs: LANGS });
  }
  return highlighterPromise;
}

/** Highlight code to a Shiki `<pre>` HTML string (github-dark-dimmed),
 *  called from getStaticProps so the highlighted markup bakes into the
 *  static export -- no client-side highlighter, no flash of unhighlighted
 *  code. */
export async function highlightCode(code: string, lang: string): Promise<string> {
  const h = await getHighlighter();
  const resolved = h.getLoadedLanguages().includes(lang) ? lang : "text";
  return h.codeToHtml(code, { lang: resolved, theme: THEME });
}

/** Highlights every block in one call (one shared highlighter instance),
 *  for a page's getStaticProps to await once and spread the results in
 *  JSX-appearance order. */
export async function highlightBlocks(
  blocks: { lang: string; code: string }[],
): Promise<string[]> {
  const h = await getHighlighter();
  return blocks.map(({ lang, code }) => {
    const resolved = h.getLoadedLanguages().includes(lang) ? lang : "text";
    return h.codeToHtml(code, { lang: resolved, theme: THEME });
  });
}
