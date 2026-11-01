import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Uploads go through the /api/predict route handler rather than a server
  // action, so no body-size experiment is needed here.
  outputFileTracingRoot: __dirname,
};

export default nextConfig;
