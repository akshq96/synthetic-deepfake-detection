import { expect, test } from "@playwright/test";
import path from "path";

const TEST_IMAGE = path.join(__dirname, "fixtures", "test-image.jpg");

test.describe("navigation", () => {
  test("dashboard loads and sidebar links to every route", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Dashboard" })).toBeVisible();
    // Scoped to the sidebar <nav> — "Synthetic Data Lab" also appears in a
    // dashboard quick-link card, which would otherwise make this ambiguous.
    const sidebarNav = page.getByRole("navigation");
    await expect(sidebarNav.getByRole("link", { name: "Image / Video Detection" })).toBeVisible();
    await expect(sidebarNav.getByRole("link", { name: "Synthetic Data Lab", exact: true })).toBeVisible();
  });

  test("detect page renders the upload widget", async ({ page }) => {
    await page.goto("/detect");
    await expect(page.getByRole("heading", { name: "Detection" })).toBeVisible();
    await expect(page.getByText("Upload an image (JPG, PNG, WEBP)")).toBeVisible();
  });
});

// Requires a running backend (http://localhost:8000 by default) with
// DEFAULT_CHECKPOINT_PATH configured — see docs/reproducibility.md. Skipped
// automatically if the backend isn't reachable, so `npm run test:e2e` still
// passes on a fresh clone with no backend running (only the navigation
// tests above run in that case).
test.describe("detect happy path (requires a running, configured backend)", () => {
  test.beforeEach(async ({ request }) => {
    const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
    let reachable = false;
    try {
      const res = await request.get(`${apiBase}/api/health`, { timeout: 2000 });
      reachable = res.ok();
    } catch {
      reachable = false;
    }
    test.skip(!reachable, "backend not reachable at " + apiBase + " — skipping live-detection e2e test");
  });

  test("uploading an image produces a prediction and a heatmap", async ({ page }) => {
    await page.goto("/detect");

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Prediction card appears with a real/fake/abstain badge. Scoped to the
    // Badge component's shared `label-mono` class — the gauge's axis label
    // also renders the literal text "Real", so a plain text query is
    // ambiguous.
    await expect(page.getByRole("heading", { name: "Prediction" })).toBeVisible({ timeout: 20_000 });
    const badge = page.locator(".label-mono").filter({ hasText: /^(Real|Fake|Uncertain \/ Abstained)$/ }).first();
    await expect(badge).toBeVisible();
    await expect(page.getByText(/% confidence/)).toBeVisible();

    // Explainability heatmap image loaded (not the "No heatmap available" fallback).
    await expect(page.getByText("No heatmap available")).not.toBeVisible();
    await expect(page.locator('img[alt="Explainability heatmap"]')).toBeVisible();
  });
});
