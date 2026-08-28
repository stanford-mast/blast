// Docs sidebar registry. Deliberately small and hand-authored (no MDX/content
// pipeline): three groups, mirroring arker.ai/docs's "Specs / Guides / API"
// pattern (cloudflare/src/lib/docs/pages.ts) but trimmed to exactly what
// BLAST ships.

export type NavItem = {
  slug: string;
  label: string;
  // Items sharing a `row` key render side-by-side in one grid row (used for
  // the Fork/Run/Sync trio, same convention as arker.ai/docs's NAV_GROUPS).
  row?: string;
};

export type NavGroup = {
  label: string;
  items: NavItem[];
};

// `slug` maps to /docs/<slug> ("" = /docs).
export const NAV_GROUPS: NavGroup[] = [
  {
    label: "Specs",
    items: [{ slug: "", label: "Introduction" }],
  },
  {
    label: "Guides",
    items: [
      { slug: "quickstart", label: "Quickstart" },
      { slug: "orchestration", label: "Orchestration" },
    ],
  },
  {
    label: "API",
    items: [
      { slug: "api", label: "Overview" },
      { slug: "api/vms", label: "VMs" },
      { slug: "api/sessions", label: "Sessions" },
      { slug: "api/fork", label: "Fork", row: "methods" },
      { slug: "api/runs", label: "Run", row: "methods" },
      { slug: "api/sync", label: "Sync", row: "methods" },
    ],
  },
];

export function hrefForSlug(slug: string): string {
  return slug ? `/docs/${slug}` : "/docs";
}
