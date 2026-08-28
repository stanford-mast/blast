// Anchor id for an API operation heading, e.g. ("GET", "/v1/vms/{run_id}") ->
// "get-v1-vms-run-id". Shared by <Endpoint as="h2"> (which sets this as its
// id) and the hand-written on-this-page / cross-page links that point at it,
// so the two can never drift apart. Underscored path params (`{run_id}`) are
// hyphenated too, matching every hand-written href in this docs set.
export function endpointSlug(method: string, path: string): string {
  const clean = path.replace(/\{([^}]+)\}/g, "$1").replace(/^\//, "");
  return `${method.toLowerCase()}-${clean.replace(/[/_]/g, "-")}`;
}
