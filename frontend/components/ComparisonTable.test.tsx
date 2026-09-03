import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ComparisonTable, type ComparisonColumn } from "./ComparisonTable";

interface Row {
  id: string;
  name: string;
  accuracy: number;
}

const columns: ComparisonColumn<Row>[] = [
  { key: "name", label: "Name" },
  { key: "accuracy", label: "Accuracy", format: (v) => Number(v).toFixed(2) },
];

const rows: Row[] = [
  { id: "a", name: "CNN", accuracy: 0.7 },
  { id: "b", name: "ViT", accuracy: 0.9 },
  { id: "c", name: "Baseline", accuracy: 0.5 },
];

describe("ComparisonTable", () => {
  it("renders every row with formatted values", () => {
    render(<ComparisonTable rows={rows} columns={columns} />);
    expect(screen.getByText("CNN")).toBeInTheDocument();
    expect(screen.getByText("0.70")).toBeInTheDocument();
    expect(screen.getByText("0.90")).toBeInTheDocument();
  });

  it("shows an empty state with no rows", () => {
    render(<ComparisonTable rows={[]} columns={columns} />);
    expect(screen.getByText("No data yet")).toBeInTheDocument();
  });

  it("sorts descending by default when a column header is clicked, then ascending on a second click", () => {
    render(<ComparisonTable rows={rows} columns={columns} />);
    const accuracyHeader = screen.getByText("Accuracy");

    fireEvent.click(accuracyHeader);
    let cells = screen.getAllByRole("cell").map((c) => c.textContent);
    // Every row has 2 cells (name, accuracy) -> accuracy cells are odd indices.
    expect([cells[1], cells[3], cells[5]]).toEqual(["0.90", "0.70", "0.50"]);

    fireEvent.click(accuracyHeader);
    cells = screen.getAllByRole("cell").map((c) => c.textContent);
    expect([cells[1], cells[3], cells[5]]).toEqual(["0.50", "0.70", "0.90"]);
  });
});
