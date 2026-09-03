import { cn } from "@/lib/cn";

export function Table({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <div className="w-full overflow-x-auto rounded-lg border border-border">
      <table className={cn("w-full text-[13px]", className)}>{children}</table>
    </div>
  );
}

export function THead({ children }: { children: React.ReactNode }) {
  return (
    <thead className="meta-label border-b border-border bg-muted/60 text-left text-[10.5px] font-medium">
      {children}
    </thead>
  );
}

export function TBody({ children }: { children: React.ReactNode }) {
  return <tbody className="divide-y divide-border bg-card">{children}</tbody>;
}

export function TR({
  className,
  children,
  onClick,
}: {
  className?: string;
  children: React.ReactNode;
  onClick?: () => void;
}) {
  return (
    <tr
      onClick={onClick}
      className={cn("transition-colors hover:bg-muted/50", onClick && "cursor-pointer", className)}
    >
      {children}
    </tr>
  );
}

export function TH({ className, children }: { className?: string; children?: React.ReactNode }) {
  return <th className={cn("whitespace-nowrap px-3.5 py-2.5", className)}>{children}</th>;
}

export function TD({ className, children }: { className?: string; children: React.ReactNode }) {
  return <td className={cn("px-3.5 py-2.5 align-middle", className)}>{children}</td>;
}
