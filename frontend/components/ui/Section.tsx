import { cn } from "@/lib/cn";

// The default way to group content in this redesign — a heading + optional
// description + divider, no card chrome. Reach for <Card> only when content
// genuinely needs visual containment (e.g. a status panel), not by default.
export function Section({
  title,
  description,
  action,
  className,
  children,
}: {
  title?: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <section className={cn("py-6 first:pt-0", className)}>
      {(title || action) && (
        <div className="mb-4 flex items-end justify-between gap-4 border-b border-border pb-3">
          <div>
            {title && <h2 className="font-display text-[15px] font-semibold tracking-tight">{title}</h2>}
            {description && <p className="mt-0.5 text-[12.5px] text-muted-foreground">{description}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}
