"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "@/components/QueryState";
import { ExperimentStatusBadge } from "@/components/ExperimentStatusBadge";
import { api } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { FlaskConical, ScanSearch, Sparkles } from "lucide-react";

export default function DashboardPage() {
  const experimentsQuery = useQuery({
    queryKey: ["experiments"],
    queryFn: api.listExperiments,
  });

  const recentRuns =
    experimentsQuery.data
      ?.flatMap((exp) => exp.runs.map((run) => ({ ...run, experimentName: exp.name })))
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
      .slice(0, 6) ?? [];

  return (
    <div>
      <PageHeader
        title="Dashboard"
        description="Synthetic Data-Augmented Deepfake Detection — research overview"
      />

      <div className="mb-6 grid gap-4 sm:grid-cols-3">
        <QuickLinkCard
          href="/detect"
          icon={ScanSearch}
          title="Run detection"
          description="Upload an image or video and inspect the model's prediction and heatmap."
        />
        <QuickLinkCard
          href="/synthetic-lab"
          icon={Sparkles}
          title="Synthetic Data Lab"
          description="Configure a synthetic-augmentation ratio/technique mix and launch a training run."
        />
        <QuickLinkCard
          href="/results/generalization"
          icon={FlaskConical}
          title="Research question"
          description="Does synthetic augmentation improve generalization to unseen manipulations?"
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent experiment runs</CardTitle>
          <CardDescription>The most recently created training/experiment runs.</CardDescription>
        </CardHeader>
        <CardContent>
          {experimentsQuery.isLoading && <LoadingState />}
          {experimentsQuery.isError && <ErrorState error={experimentsQuery.error} />}
          {experimentsQuery.isSuccess && recentRuns.length === 0 && (
            <EmptyState label="No experiment runs yet — launch one from the Synthetic Data Lab." />
          )}
          {recentRuns.length > 0 && (
            <ul className="divide-y divide-border">
              {recentRuns.map((run) => (
                <li key={run.id} className="flex items-center justify-between py-3 text-sm">
                  <div>
                    <div className="font-medium">{run.run_name}</div>
                    <div className="text-xs text-muted-foreground">{run.experimentName}</div>
                  </div>
                  <ExperimentStatusBadge status={run.status} />
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function QuickLinkCard({
  href,
  icon: Icon,
  title,
  description,
}: {
  href: string;
  icon: React.ElementType;
  title: string;
  description: string;
}) {
  return (
    <Link href={href}>
      <Card className="h-full transition-colors hover:border-primary/50">
        <CardContent className="pt-5">
          <Icon className="mb-3 h-5 w-5 text-primary" />
          <div className="mb-1 text-sm font-semibold">{title}</div>
          <div className="text-xs text-muted-foreground">{description}</div>
        </CardContent>
      </Card>
    </Link>
  );
}
