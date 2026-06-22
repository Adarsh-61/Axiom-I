/** @type {import('next').NextConfig} */

// NOTE: If the backend URL (NEXT_PUBLIC_API_BASE) changes, you must also update
// the `connect-src` directive in src/app/middleware.ts to prevent the browser
// from blocking API requests due to Content-Security-Policy violations.
const nextConfig = {};

export default nextConfig;
