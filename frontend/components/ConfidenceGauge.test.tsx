import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ConfidenceGauge } from "./ConfidenceGauge";

function getVerdict() {
  return document.querySelector('[data-slot="verdict"]');
}

describe("ConfidenceGauge", () => {
  it("renders the Deepfake verdict and confidence for a fake prediction", () => {
    render(<ConfidenceGauge label="fake" confidence={0.913} fakeProbability={0.913} />);
    expect(getVerdict()).toHaveTextContent("Deepfake");
    expect(screen.getByText("91.3% confidence")).toBeInTheDocument();
    // Both probability rows are always shown, regardless of verdict.
    expect(screen.getByText("91.3%")).toBeInTheDocument();
    expect(screen.getByText("8.7%")).toBeInTheDocument();
  });

  it("renders the Real verdict for a real prediction", () => {
    render(<ConfidenceGauge label="real" confidence={0.87} fakeProbability={0.13} />);
    expect(getVerdict()).toHaveTextContent("Real");
    expect(screen.getByText("87.0% confidence")).toBeInTheDocument();
    expect(screen.getByText("13.0%")).toBeInTheDocument();
  });

  it("renders the Uncertain verdict distinctly", () => {
    render(<ConfidenceGauge label="abstain" confidence={0.52} fakeProbability={0.52} />);
    expect(getVerdict()).toHaveTextContent("Uncertain");
  });
});
