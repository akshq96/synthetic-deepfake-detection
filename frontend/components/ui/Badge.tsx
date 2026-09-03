import { cn } from "@/lib/cn";

type BadgeVariant = "default" | "success" | "danger" | "warning" | "muted";

const VARIANT_CLASSES: Record<BadgeVariant, string> = {
  default: "bg-primary-tint text-primary",
  success: "bg-success-tint text-success",
  danger: "bg-danger-tint text-danger",
  warning: "bg-warning-tint text-warning",
  muted: "bg-muted text-muted-foreground",
};

const DOT_CLASSES: Record<BadgeVariant, string> = {
  default: "bg-primary",
  success: "bg-success",
  danger: "bg-danger",
  warning: "bg-warning",
  muted: "bg-muted-foreground",
};

// A quiet tinted label with a status dot — used for verdicts/statuses only,
// not decoratively. Prefer plain text + color for one-off metadata instead
// of reaching for a badge every time.
export function Badge({
  variant = "default",
  dot = true,
  className,
  children,
}: {
  variant?: BadgeVariant;
  dot?: boolean;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <span
      data-slot="badge"
      className={cn(
        "inline-flex items-center gap-1.5 rounded px-2 py-0.5 text-[11px] font-medium",
        VARIANT_CLASSES[variant],
        className
      )}
    >
      {dot && <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", DOT_CLASSES[variant])} />}
      {children}
    </span>
  );
}
