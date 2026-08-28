import Link from "next/link";
import { useRouter } from "next/router";
import { NAV_GROUPS, hrefForSlug, type NavItem } from "@/lib/docs/nav";

// Left sidebar. No border, no surface tint on the rail itself; rows touch (no
// gaps), small uppercase-ish section labels, active row gets a subtle fill.
// Mirrors arker.ai/docs's DocsNav (cloudflare/src/components/docs/docs-nav.tsx),
// ported from next/navigation to next/router for the pages-router site.
export function DocsNav({ onNavigate }: { onNavigate?: () => void }) {
  const router = useRouter();

  const link = (item: NavItem) => {
    const href = hrefForSlug(item.slug);
    const active = router.pathname === href;
    return (
      <Link
        key={item.slug}
        href={href}
        onClick={onNavigate}
        className={
          "block px-3 py-1.5 transition-colors duration-50 " +
          (active
            ? "bg-[#2e2e2e] text-ink"
            : "text-ink-secondary hover:bg-surface hover:text-ink")
        }
      >
        {item.label}
      </Link>
    );
  };

  return (
    <nav aria-label="Documentation" className="flex flex-col gap-6 text-sm">
      {NAV_GROUPS.map((group) => (
        <div key={group.label}>
          <p className="mb-1 px-3 text-xs font-medium text-ink-tertiary">
            {group.label}
          </p>
          <div className="flex flex-col">
            {clusterByRow(group.items).map((cluster, ci) =>
              cluster.length > 1 ? (
                // No gap; rows touch, matching arker.ai/docs's DocsNav
                // (docs-nav.tsx:41-59), which sizes each clustered row with
                // an explicit `repeat(n, minmax(0, 1fr))` grid rather than a
                // fixed column count.
                <div
                  key={ci}
                  className="grid"
                  style={{
                    gridTemplateColumns: `repeat(${cluster.length}, minmax(0, 1fr))`,
                  }}
                >
                  {cluster.map((item) => (
                    <div key={item.slug}>{link(item)}</div>
                  ))}
                </div>
              ) : (
                <div key={ci}>{link(cluster[0])}</div>
              ),
            )}
          </div>
        </div>
      ))}
    </nav>
  );
}

// Group consecutive items sharing a `row` key into one cluster (rendered as a
// grid row); every other item is its own single-element cluster.
function clusterByRow(items: NavItem[]): NavItem[][] {
  const out: NavItem[][] = [];
  for (const item of items) {
    const last = out[out.length - 1];
    if (item.row && last && last[0]?.row === item.row) last.push(item);
    else out.push([item]);
  }
  return out;
}
