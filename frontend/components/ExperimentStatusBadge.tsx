import { Badge } from "@/components/ui/Badge";
import { Spinner } from "@/components/ui/Spinner";

const STATUS_META: Record<string, { text: string; variant: "success" | "danger" | "warning" | "muted" }> = {
  pending: { text: "Pending", variant: "muted" },
  running: { text: "Running", variant: "warning" },
  completed: { text: "Completed", variant: "success" },
  failed: { text: "Failed", variant: "danger" },
};

export function ExperimentStatusBadge({ status }: { status: string }) {
  const meta = STATUS_META[status] ?? { text: status, variant: "muted" as const };
  return (
    <Badge variant={meta.variant}>
      {status === "running" && <Spinner className="h-3 w-3" />}
      {meta.text}
    </Badge>
  );
}
