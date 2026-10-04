import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The Python server (biovision serve) runs on port 8000; the dev server proxies to it.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  test: { environment: "node" },
});
