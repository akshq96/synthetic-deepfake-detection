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
        <div className="group relative overflow-hidden rounded-md border border-primary/25 bg-primary/[0.04] px-6 py-5 transition-colors hover:border-primary/50">
          <div className="label-mono mb-2 text-[10px] text-primary">Primary Research Question</div>
          <p className="font-display max-w-2xl text-[17px] leading-snug font-medium tracking-tight">
            Does synthetic data augmentation improve generalization of deepfake detectors to unseen
            manipulation techniques?
          </p>
          <div className="mt-3 inline-flex items-center gap-1 text-[12px] text-primary opacity-80 transition-opacity group-hover:opacity-100">
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
      <Card className="h-full transition-colors hover:border-primary/40">
        <CardContent className="flex items-start gap-3.5 pt-5">
          <span className="label-mono mt-0.5 text-[10px] text-muted-foreground/60">{index}</span>
          <div className="flex-1">
            <Icon className="mb-2.5 h-[18px] w-[18px] text-primary" strokeWidth={1.75} />
            <div className="font-display mb-1 text-[13.5px] font-semibold">{title}</div>
            <div className="text-xs leading-relaxed text-muted-foreground">{description}</div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
