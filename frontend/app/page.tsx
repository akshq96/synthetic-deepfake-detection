"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "@/components/QueryState";
import { ExperimentStatusBadge } from "@/components/ExperimentStatusBadge";
import { api } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { ArrowUpRight, ScanSearch, Sparkles } from "lucide-react";

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
        eyebrow="Overview"
        title="Dashboard"
        description="Synthetic Data-Augmented Deepfake Detection — a research system for whether training on real + synthetically-manipulated data improves generalization to manipulation patterns never seen during training."
      />

      <Link href="/results/generalization" className="mb-6 block">
        <div className="glow-ring group relative overflow-hidden rounded-xl bg-gradient-to-br from-primary/10 via-card to-primary-2/[0.06] px-6 py-6 transition-transform duration-200 hover:-translate-y-0.5">
          <div className="pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full bg-primary-2/20 blur-3xl" />
          <div className="label-mono mb-2 text-[10px] text-primary">Primary Research Question</div>
          <p className="font-display relative max-w-2xl text-[19px] leading-snug font-medium tracking-tight">
            Does synthetic data augmentation improve generalization of deepfake detectors to unseen
            manipulation techniques?
          </p>
          <div className="relative mt-3 inline-flex items-center gap-1 text-[12px] font-medium text-primary opacity-90 transition-opacity group-hover:opacity-100">
            View generalization results
            <ArrowUpRight className="h-3.5 w-3.5" />
          </div>
        </div>
      </Link>

      <div className="mb-6 grid gap-4 sm:grid-cols-2">
        <QuickLinkCard
          index="01"
          href="/detect"
          icon={ScanSearch}
          title="Run detection"
          description="Upload an image or video and inspect the model's prediction and heatmap."
        />
        <QuickLinkCard
          index="02"
          href="/synthetic-lab"
          icon={Sparkles}
          title="Synthetic Data Lab"
          description="Configure a synthetic-augmentation ratio/technique mix and launch a training run."
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
  index,
  href,
  icon: Icon,
  title,
  description,
}: {
  index: string;
  href: string;
  icon: React.ElementType;
  title: string;
  description: string;
}) {
  return (
    <Link href={href}>
      <Card className="h-full transition-all duration-200 hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-[0_12px_30px_-14px_var(--primary)]">
        <CardContent className="flex items-start gap-3.5 pt-5">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-primary/20 to-primary-2/10">
            <Icon className="h-[17px] w-[17px] text-primary" strokeWidth={1.75} />
          </div>
          <div className="flex-1">
            <div className="mb-1 flex items-center gap-2">
              <div className="font-display text-[13.5px] font-semibold">{title}</div>
              <span className="label-mono text-[9px] text-muted-foreground/50">{index}</span>
            </div>
            <div className="text-xs leading-relaxed text-muted-foreground">{description}</div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
