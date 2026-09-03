import { cn } from "@/lib/cn";

export function Table({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <div className="w-full overflow-x-auto">
      <table className={cn("w-full text-sm", className)}>{children}</table>
    </div>
  );
}

export function THead({ children }: { children: React.ReactNode }) {
  return <thead className="border-b border-border text-left text-xs text-muted-foreground">{children}</thead>;
}

export function TBody({ children }: { children: React.ReactNode }) {
  return <tbody className="divide-y divide-border">{children}</tbody>;
}

export function TR({ className, children }: { className?: string; children: React.ReactNode }) {
  return <tr className={cn("hover:bg-muted/50", className)}>{children}</tr>;
}

export function TH({ className, children }: { className?: string; children: React.ReactNode }) {
  return <th className={cn("px-3 py-2 font-medium", className)}>{children}</th>;
}

export function TD({ className, children }: { className?: string; children: React.ReactNode }) {
  return <td className={cn("px-3 py-2", className)}>{children}</td>;
}
