/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',  // Enable static exports
  basePath: '',      // Set this if deploying to a subdirectory
  images: {
    unoptimized: true,  // Required for static export
  },
}

module.exports = nextConfig