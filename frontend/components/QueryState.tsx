import { Button } from "@/components/ui/Button";
import { ApiError } from "@/lib/api-client";
import { AlertTriangle, Inbox, Loader2 } from "lucide-react";

export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2.5 py-16 text-[13px] text-muted-foreground">
      <Loader2 className="h-4 w-4 animate-spin" />
      {label}
    </div>
  );
}

/**
 * A connectivity/API failure — distinct from EmptyState (which means "the
 * request worked, there's just nothing there yet"). Explains what failed,
 * offers a likely reason, and a retry action, per the redesign brief:
 * "Failed to fetch" alone reads like a broken dev screen.
 */
export function ErrorState({
  error,
  title = "Unable to load this data",
  onRetry,
}: {
  error: unknown;
  title?: string;
  onRetry?: () => void;
}) {
  const isConnectivity =
    error instanceof TypeError ||
    (error instanceof Error && /fetch|network/i.test(error.message));

  const detail =
    error instanceof ApiError
      ? typeof error.detail === "string"
        ? error.detail
        : error.message
      : error instanceof Error
        ? error.message
        : "An unknown error occurred.";

  return (
    <div className="flex flex-col items-start gap-3 rounded-lg border border-border bg-muted/40 px-5 py-5">
      <div className="flex items-center gap-2.5">
        <AlertTriangle className="h-4 w-4 shrink-0 text-danger" />
        <span className="text-[13.5px] font-medium">{title}</span>
      </div>
      <p className="max-w-md text-[12.5px] leading-relaxed text-muted-foreground">
        {isConnectivity
          ? "Connect the analysis service and try again — the backend API may not be running."
          : detail}
      </p>
      {onRetry && (
        <Button variant="secondary" onClick={onRetry} className="mt-1">
          Retry
        </Button>
      )}
    </div>
  );
}

export function EmptyState({
  label,
  title,
  description,
  action,
}: {
  /** Simple form: one line of muted text (kept for lightweight call sites). */
  label?: string;
  /** Richer form: heading + description + optional action. */
  title?: string;
  description?: string;
  action?: React.ReactNode;
}) {
  if (title) {
    return (
      <div className="flex flex-col items-start gap-3 rounded-lg border border-dashed border-border px-6 py-10">
        <Inbox className="h-5 w-5 text-muted-foreground/60" strokeWidth={1.5} />
        <div>
          <div className="text-[14px] font-medium">{title}</div>
          {description && (
            <p className="mt-1 max-w-md text-[12.5px] leading-relaxed text-muted-foreground">{description}</p>
          )}
        </div>
        {action}
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-dashed border-border px-6 py-8 text-center text-[13px] text-muted-foreground">
      {label}
    </div>
  );
}
