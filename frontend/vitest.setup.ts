import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// @testing-library/react's auto-cleanup only self-registers when it detects
// Jest's or Vitest's global `afterEach` (via `test.globals: true`), which
// this project doesn't enable (to keep vitest's API explicitly imported
// rather than ambient-global) — so register it here instead.
afterEach(() => {
  cleanup();
});
