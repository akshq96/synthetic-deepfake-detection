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
        "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-10 text-center transition-colors",
        isDragOver ? "border-primary bg-primary/5" : "border-border",
        disabled && "cursor-not-allowed opacity-60"
      )}
    >
      <UploadCloud className="h-8 w-8 text-muted-foreground" />
      <div className="text-sm font-medium">{label}</div>
      <div className="text-xs text-muted-foreground">Drag & drop, or click to browse</div>
      {selectedFile && (
        <div className="mt-2 rounded-md bg-muted px-2.5 py-1 text-xs">{selectedFile.name}</div>
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
