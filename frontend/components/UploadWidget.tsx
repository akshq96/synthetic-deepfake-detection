"use client";

import { cn } from "@/lib/cn";
import { UploadCloud } from "lucide-react";
import { useCallback, useRef, useState } from "react";

export function UploadWidget({
  accept,
  onFileSelected,
  disabled,
  label,
  hint,
}: {
  accept: string;
  onFileSelected: (file: File) => void;
  disabled?: boolean;
  label: string;
  hint?: string;
}) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (!file) return;
      setSelectedFile(file);
      onFileSelected(file);
    },
    [onFileSelected]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragOver(true);
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragOver(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-lg border border-dashed p-8 text-center transition-all duration-200",
        isDragOver ? "scale-[1.01] border-primary bg-primary-tint" : "border-border-strong hover:bg-muted/60",
        disabled && "cursor-not-allowed opacity-60"
      )}
    >
      <UploadCloud
        className={cn(
          "mb-1.5 h-6 w-6 text-muted-foreground transition-transform duration-200",
          isDragOver && "-translate-y-0.5"
        )}
        strokeWidth={1.5}
      />
      <div className="text-[13.5px] font-medium">{label}</div>
      <div className="text-[12px] text-muted-foreground">Drag and drop, or click to browse</div>
      {hint && <div className="mt-2 text-[11px] text-muted-foreground/80">{hint}</div>}
      {selectedFile && (
        <div className="mono-value mt-3 rounded border border-border bg-card px-2.5 py-1 text-[11px]">
          {selectedFile.name}
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        disabled={disabled}
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}
