export function BrandMark({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className="shrink-0">
      <path
        d="M12 2.5 4 5.5v6c0 5 3.4 8.6 8 10 4.6-1.4 8-5 8-10v-6l-8-3Z"
        stroke="var(--primary)"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <path
        d="M8.3 12.2 11 15l4.8-5.6"
        stroke="var(--primary)"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={className}>
      <span className="font-display block text-[15px] font-semibold leading-none tracking-tight">
        AegisTrace
      </span>
      <span className="mt-1 block text-[10.5px] leading-none text-muted-foreground">Deepfake Forensics</span>
    </span>
  );
}
