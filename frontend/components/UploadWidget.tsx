"use client";

import { cn } from "@/lib/cn";
import { UploadCloud } from "lucide-react";
import { useCallback, useRef, useState } from "react";

export function UploadWidget({
  accept,
  onFileSelected,
  disabled,
  label,
}: {
  accept: string;
  onFileSelected: (file: File) => void;
  disabled?: boolean;
  label: string;
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
        "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed p-10 text-center transition-all",
        isDragOver
          ? "border-primary bg-gradient-to-br from-primary/10 to-primary-2/5 shadow-[0_0_0_4px_color-mix(in_srgb,var(--primary)_12%,transparent)]"
          : "border-border hover:border-primary/40 hover:bg-muted/40",
        disabled && "cursor-not-allowed opacity-60"
      )}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-primary/15 to-primary-2/10">
        <UploadCloud className="h-5 w-5 text-primary" strokeWidth={1.75} />
      </div>
      <div className="mt-1 text-sm font-medium">{label}</div>
      <div className="text-xs text-muted-foreground">Drag & drop, or click to browse</div>
      {selectedFile && (
        <div className="label-mono mt-2 rounded-full bg-muted px-3 py-1 text-[10px]">{selectedFile.name}</div>
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
