"use client";

import { useEffect, useRef, useState } from "react";
import type { TocItem } from "./DocsLayout";

// Ports arker.ai/docs's OnThisPage (cloudflare/src/components/docs/on-this-page.tsx)
// as closely as BLAST's flat (no nested sub-items) toc needs: the active row
// is driven by BOTH the section scrolled into view (scroll-spy) and the
// section the cursor is hovering inside the article, same "reading along"
// behavior as the reference. Hovering a row in this list only gives it a
// subtle hover background; it does not move the active row. Clicking
// smooth-scrolls from the current position. Dropped relative to the
// reference: the "dated" changelog-row split and nested sub-item groups,
// neither of which BLAST's docs have a use for.
export function OnThisPage({
  items,
  articleSelector = ".doc-content",
}: {
  items: TocItem[];
  articleSelector?: string;
}) {
  const [active, setActive] = useState<string | null>(items[0]?.id ?? null);
  // True only when the active row last moved because the cursor was hovering
  // over the article content (the "reading along" case), that highlight
  // glides a touch slower. Rail hover, clicks, and scroll-spy stay snappy.
  const [slow, setSlow] = useState(false);
  // While set, the active row is LOCKED to this id, used during a
  // click-driven smooth scroll so the highlight doesn't flicker through
  // every section it passes. Released when the scroll settles or the
  // cursor moves over content.
  const lockedRef = useRef<string | null>(null);
  const lockTimeoutRef = useRef<number>(0);

  useEffect(() => {
    if (!items.length) return;
    const headings = items
      .map((i) => document.getElementById(i.id))
      .filter((el): el is HTMLElement => el != null);
    if (!headings.length) return;

    const article = document.querySelector<HTMLElement>(articleSelector);
    // Reference line: the cursor's Y while it's over the article, else a
    // band just under the top nav (scroll-spy). The active row is the last
    // heading at/above that line. Recomputed on both mousemove and scroll,
    // so scrolling with the mouse held still re-evaluates the section now
    // under the cursor.
    const BAND = 130;
    let mouseY = -1; // -1 = cursor not over the article
    let raf = 0;
    let settleTimer = 0;
    const recompute = () => {
      raf = 0;
      if (lockedRef.current) {
        setActive(lockedRef.current);
        return;
      }
      const y = mouseY >= 0 ? mouseY : BAND;
      let current = headings[0].id;
      for (const h of headings) {
        if (h.getBoundingClientRect().top - 8 <= y) current = h.id;
        else break;
      }
      setActive(current);
      setSlow(mouseY >= 0);
    };
    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(recompute);
    };
    const onScroll = () => {
      schedule();
      if (lockedRef.current) {
        clearTimeout(settleTimer);
        settleTimer = window.setTimeout(() => {
          lockedRef.current = null;
        }, 120);
      }
    };
    const onMove = (e: MouseEvent) => {
      mouseY = e.clientY;
      if (lockedRef.current) {
        lockedRef.current = null;
        clearTimeout(settleTimer);
        clearTimeout(lockTimeoutRef.current);
      }
      schedule();
    };
    const onLeave = () => {
      mouseY = -1;
      setSlow(false);
      if (raf) {
        cancelAnimationFrame(raf);
        raf = 0;
      }
    };
    article?.addEventListener("mousemove", onMove);
    article?.addEventListener("mouseleave", onLeave);
    window.addEventListener("scroll", onScroll, { passive: true });
    recompute();

    return () => {
      article?.removeEventListener("mousemove", onMove);
      article?.removeEventListener("mouseleave", onLeave);
      window.removeEventListener("scroll", onScroll);
      clearTimeout(settleTimer);
      if (raf) cancelAnimationFrame(raf);
    };
  }, [items, articleSelector]);

  if (!items.length) return null;

  const jump = (e: React.MouseEvent, id: string) => {
    e.preventDefault();
    const el = document.getElementById(id);
    if (!el) return;
    lockedRef.current = id;
    clearTimeout(lockTimeoutRef.current);
    lockTimeoutRef.current = window.setTimeout(() => {
      lockedRef.current = null;
    }, 1000);
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    history.replaceState(null, "", `#${id}`);
    setSlow(false);
    setActive(id);
  };

  return (
    <nav aria-label="On this page" className="flex flex-col text-sm">
      {items.map((item) => {
        const on = active === item.id;
        return (
          <a
            key={item.id}
            href={`#${item.id}`}
            onClick={(e) => jump(e, item.id)}
            style={{ transitionDuration: slow ? "150ms" : "50ms" }}
            className={
              "block rounded-none px-2 py-2 leading-snug transition-colors " +
              (on
                ? "bg-[#2e2e2e] text-ink"
                : "text-ink-tertiary hover:bg-surface hover:text-ink")
            }
          >
            {item.text}
          </a>
        );
      })}
    </nav>
  );
}
