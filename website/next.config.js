/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',  // Enable static exports
  basePath: '',      // Set this if deploying to a subdirectory
  images: {
    unoptimized: true,  // Required for static export
  },
  // The docs section adds nested routes (e.g. /docs/api/vms). trailingSlash
  // makes `next export` emit a `<route>/index.html` for every one of them
  // instead of a flat `<route>.html` sibling file — the directory+index.html
  // shape every static host (GitHub Pages included) resolves for a clean-URL
  // request, without depending on host-specific ".html" extension guessing.
  trailingSlash: true,
}

module.exports = nextConfig
