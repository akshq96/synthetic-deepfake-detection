import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ExperimentStatusBadge } from "./ExperimentStatusBadge";

describe("ExperimentStatusBadge", () => {
  it.each([
    ["pending", "Pending"],
    ["running", "Running"],
    ["completed", "Completed"],
    ["failed", "Failed"],
  ])("renders the correct label for status %s", (status, label) => {
    render(<ExperimentStatusBadge status={status} />);
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it("falls back to the raw status string for an unknown status", () => {
    render(<ExperimentStatusBadge status="weird_status" />);
    expect(screen.getByText("weird_status")).toBeInTheDocument();
  });
});
