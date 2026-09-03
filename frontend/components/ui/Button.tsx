import { cn } from "@/lib/cn";
import { ButtonHTMLAttributes } from "react";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary: "border border-primary bg-primary text-primary-foreground hover:bg-primary/90",
  secondary: "border border-border bg-muted text-foreground hover:border-primary/40",
  ghost: "border border-transparent bg-transparent text-foreground hover:bg-muted",
  danger: "border border-danger bg-danger text-danger-foreground hover:bg-danger/90",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
}

export function Button({ variant = "primary", className, children, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-md px-3.5 py-2 text-[13px] font-medium",
        "transition-colors disabled:opacity-50 disabled:pointer-events-none",
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
