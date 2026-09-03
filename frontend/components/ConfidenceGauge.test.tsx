import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ConfidenceGauge } from "./ConfidenceGauge";

describe("ConfidenceGauge", () => {
  it("renders the Fake badge and confidence percentage for a fake prediction", () => {
    render(<ConfidenceGauge label="fake" confidence={0.913} fakeProbability={0.913} />);
    expect(screen.getByText("Fake")).toBeInTheDocument();
    expect(screen.getByText("91.3% confidence")).toBeInTheDocument();
    expect(screen.getByText("Fake probability: 91.3%")).toBeInTheDocument();
  });

  it("renders the Real badge for a real prediction", () => {
    render(<ConfidenceGauge label="real" confidence={0.87} fakeProbability={0.13} />);
    // "Real" appears twice: the status badge and the gauge's axis label —
    // assert the badge specifically via its distinguishing class rather
    // than an ambiguous text match.
    const badge = document.querySelector(".bg-success\\/10");
    expect(badge).toHaveTextContent("Real");
    expect(screen.getByText("87.0% confidence")).toBeInTheDocument();
  });

  it("renders the abstain badge distinctly", () => {
    render(<ConfidenceGauge label="abstain" confidence={0.52} fakeProbability={0.52} />);
    expect(screen.getByText("Uncertain / Abstained")).toBeInTheDocument();
  });
});
