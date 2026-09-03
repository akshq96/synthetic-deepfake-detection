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
    <div className="mb-7 flex items-start justify-between gap-4 border-b border-border pb-5">
      <div>
        {eyebrow && <div className="label-mono mb-1.5 text-[10px] text-primary">{eyebrow}</div>}
        <h1 className="font-display text-[22px] font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1.5 max-w-2xl text-[13px] text-muted-foreground">{description}</p>}
      </div>
      {action}
    </div>
  );
}
