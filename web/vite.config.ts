import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The app bundles the Python sources straight from ../drone_sim and
// ../controllers (see src/sim/worker.ts and src/controllers.ts), so the dev
// server must be allowed to read the repo root.
const repoRoot = fileURLToPath(new URL("..", import.meta.url));

export default defineConfig({
  plugins: [react()],
  base: "./", // works from any sub-path, e.g. GitHub Pages
  server: { fs: { allow: [repoRoot] } },
  worker: { format: "es" },
  build: { chunkSizeWarningLimit: 2000 },
});
