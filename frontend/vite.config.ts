import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url))
    }
  },
  server: {
    port: 5173,
    proxy: {
      "/login": "http://localhost:8001",
      "/register": "http://localhost:8001",
      "/refresh": "http://localhost:8001",
      "/profile": "http://localhost:8001",
      "/kb": "http://localhost:8001",
      "/document": "http://localhost:8001",
      "/embedding": "http://localhost:8001",
      "/agent": "http://localhost:8001",
      "/memory": "http://localhost:8001",
      "/conversations": "http://localhost:8001",
      "/metrics": "http://localhost:8001",
      "/health": "http://localhost:8001"
    }
  }
});
