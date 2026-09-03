import { cn } from "@/lib/cn";

type BadgeVariant = "default" | "success" | "danger" | "warning" | "muted";

const VARIANT_CLASSES: Record<BadgeVariant, string> = {
  default: "bg-primary/15 text-primary shadow-[0_0_0_1px_color-mix(in_srgb,var(--primary)_25%,transparent)]",
  success: "bg-success/15 text-success shadow-[0_0_0_1px_color-mix(in_srgb,var(--success)_25%,transparent)]",
  danger: "bg-danger/15 text-danger shadow-[0_0_0_1px_color-mix(in_srgb,var(--danger)_25%,transparent)]",
  warning: "bg-warning/15 text-warning shadow-[0_0_0_1px_color-mix(in_srgb,var(--warning)_25%,transparent)]",
  muted: "bg-muted text-muted-foreground",
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
        "label-mono inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[10px] font-semibold",
        VARIANT_CLASSES[variant],
        className
      )}
    >
      {children}
    </span>
  );
}
