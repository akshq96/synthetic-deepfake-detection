import { cn } from "@/lib/cn";

type BadgeVariant = "default" | "success" | "danger" | "warning" | "muted";

const VARIANT_CLASSES: Record<BadgeVariant, string> = {
  default: "border-primary/30 bg-primary/[0.08] text-primary",
  success: "border-success/30 bg-success/[0.08] text-success",
  danger: "border-danger/30 bg-danger/[0.08] text-danger",
  warning: "border-warning/30 bg-warning/[0.08] text-warning",
  muted: "border-border bg-muted text-muted-foreground",
};

export function Badge({
  variant = "default",
  className,
  children,
}: {
  variant?: BadgeVariant;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <span
      className={cn(
        "label-mono inline-flex items-center gap-1.5 rounded-sm border px-2 py-0.5 text-[10px] font-medium",
        VARIANT_CLASSES[variant],
        className
      )}
    >
      {children}
    </span>
  );
}
