import { cn } from "@/lib/cn";
import { ButtonHTMLAttributes } from "react";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary:
    "border border-transparent bg-gradient-to-r from-primary to-primary-2 text-primary-foreground shadow-[0_4px_20px_-6px_var(--primary)] hover:brightness-110",
  secondary: "border border-border bg-muted text-foreground hover:border-primary/40",
  ghost: "border border-transparent bg-transparent text-foreground hover:bg-muted",
  danger: "border border-transparent bg-danger text-danger-foreground hover:bg-danger/90",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
}

export function Button({ variant = "primary", className, children, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-[13px] font-medium",
        "transition-all disabled:opacity-50 disabled:pointer-events-none",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40",
        VARIANT_CLASSES[variant],
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
