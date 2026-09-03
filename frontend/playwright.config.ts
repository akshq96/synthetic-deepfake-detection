import { defineConfig, devices } from "@playwright/test";

// The e2e detect-happy-path test needs a running backend with a configured
// checkpoint (DEFAULT_CHECKPOINT_PATH) — see docs/reproducibility.md. This
// config only manages the frontend dev server; start the backend yourself
// before running `npm run test:e2e`.
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: "http://localhost:3000",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:3000",
    reuseExistingServer: true,
    timeout: 30_000,
  },
});
