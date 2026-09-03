import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { UploadWidget } from "./UploadWidget";

function makeFile(name = "test.jpg", type = "image/jpeg") {
  return new File(["fake image bytes"], name, { type });
}

describe("UploadWidget", () => {
  it("calls onFileSelected and shows the filename when a file is chosen via the input", () => {
    const onFileSelected = vi.fn();
    render(<UploadWidget accept="image/*" label="Upload an image" onFileSelected={onFileSelected} />);

    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const file = makeFile();
    fireEvent.change(input, { target: { files: [file] } });

    expect(onFileSelected).toHaveBeenCalledTimes(1);
    expect(onFileSelected).toHaveBeenCalledWith(file);
    expect(screen.getByText("test.jpg")).toBeInTheDocument();
  });

  it("calls onFileSelected when a file is dropped", () => {
    const onFileSelected = vi.fn();
    render(<UploadWidget accept="image/*" label="Upload an image" onFileSelected={onFileSelected} />);

    const file = makeFile("dropped.png", "image/png");
    const dropzone = screen.getByText("Upload an image").closest("div")!.parentElement!;
    fireEvent.drop(dropzone, { dataTransfer: { files: [file] } });

    expect(onFileSelected).toHaveBeenCalledWith(file);
  });

  it("does not call onFileSelected when disabled and clicked", () => {
    const onFileSelected = vi.fn();
    render(<UploadWidget accept="image/*" label="Upload an image" onFileSelected={onFileSelected} disabled />);

    const dropzone = screen.getByText("Upload an image").closest("div")!.parentElement!;
    fireEvent.click(dropzone);
    // No file dialog can actually open in jsdom either way; the meaningful
    // assertion is that drop is ignored while disabled.
    const file = makeFile();
    fireEvent.drop(dropzone, { dataTransfer: { files: [file] } });
    expect(onFileSelected).not.toHaveBeenCalled();
  });
});
