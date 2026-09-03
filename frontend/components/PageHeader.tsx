export function PageHeader({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="mb-8 flex flex-col gap-4 border-b border-border pb-6 sm:flex-row sm:items-end sm:justify-between">
      <div>
        {eyebrow && <div className="meta-label mb-2 text-[10.5px]">{eyebrow}</div>}
        <h1 className="font-display text-[24px] font-semibold tracking-tight sm:text-[26px]">{title}</h1>
        {description && (
          <p className="mt-2 max-w-2xl text-[13.5px] leading-relaxed text-muted-foreground">{description}</p>
        )}
      </div>
      {action}
    </div>
  );
}
