/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  env: {
    BACKEND_URL: process.env.BACKEND_URL || 'http://localhost:8001',
    WS_URL: process.env.WS_URL || 'ws://localhost:8001/ws',
  },
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: 'http://localhost:8001/:path*',
      },
    ];
  },
}

module.exports = nextConfig