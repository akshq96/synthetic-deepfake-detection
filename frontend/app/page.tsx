"use client";

import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/ui/Section";
import { EmptyState, ErrorState, LoadingState } from "@/components/QueryState";
import { ExperimentStatusBadge } from "@/components/ExperimentStatusBadge";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/Table";
import { api } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { ArrowRight, FlaskConical, ScanSearch, Sparkles, ShieldCheck, Microscope } from "lucide-react";

const WORKFLOW_STEPS = [
  { icon: ScanSearch, label: "Analyze media", description: "Upload an image or video" },
  { icon: ShieldCheck, label: "Get a verdict", description: "Calibrated real / deepfake confidence" },
  { icon: Microscope, label: "Understand evidence", description: "Grad-CAM / attention heatmap" },
  { icon: FlaskConical, label: "Review research", description: "Does augmentation generalize better?" },
];

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
        description="A research system testing whether training on real footage plus controlled synthetic manipulations improves deepfake detection and its ability to generalize to manipulation patterns never seen during training."
        action={
          <Link href="/detect">
            <span className="inline-flex items-center gap-1.5 rounded-md border border-primary bg-primary px-4 py-2 text-[13px] font-medium text-primary-foreground hover:bg-primary-hover">
              Analyze media
              <ArrowRight className="h-3.5 w-3.5" />
            </span>
          </Link>
        }
      />

      <Section>
        <div className="grid gap-x-8 gap-y-6 sm:grid-cols-2 lg:grid-cols-4">
          {WORKFLOW_STEPS.map((step, i) => (
            <div key={step.label} className="relative">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border-strong bg-card">
                  <step.icon className="h-[15px] w-[15px] text-primary" strokeWidth={1.75} />
                </div>
                <span className="meta-label text-[10px]">Step {i + 1}</span>
              </div>
              <div className="mt-2.5 text-[13.5px] font-medium">{step.label}</div>
              <div className="mt-0.5 text-[12px] text-muted-foreground">{step.description}</div>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Primary research question">
        <div className="rounded-lg border border-border bg-card px-5 py-5">
          <p className="font-display max-w-2xl text-[17px] font-medium leading-snug tracking-tight">
            Does synthetic data augmentation improve generalization of deepfake detectors to unseen
            manipulation techniques?
          </p>
          <div className="mt-4 flex flex-wrap gap-3">
            <Link
              href="/synthetic-lab"
              className="inline-flex items-center gap-1.5 text-[12.5px] font-medium text-primary hover:underline"
            >
              <Sparkles className="h-3.5 w-3.5" />
              Configure synthetic data
            </Link>
            <span className="text-border-strong">·</span>
            <Link
              href="/results/generalization"
              className="inline-flex items-center gap-1.5 text-[12.5px] font-medium text-primary hover:underline"
            >
              View generalization results
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>
      </Section>

      <Section
        title="Recent experiment runs"
        description="The most recently created training/experiment runs, launched from the Synthetic Data Lab."
      >
        {experimentsQuery.isLoading && <LoadingState />}
        {experimentsQuery.isError && (
          <ErrorState error={experimentsQuery.error} onRetry={() => experimentsQuery.refetch()} />
        )}
        {experimentsQuery.isSuccess && recentRuns.length === 0 && (
          <EmptyState
            title="No experiments recorded"
            description="Training and evaluation runs will appear here once you launch a research experiment."
            action={
              <Link href="/synthetic-lab" className="text-[12.5px] font-medium text-primary hover:underline">
                Open Synthetic Data Lab →
              </Link>
            }
          />
        )}
        {recentRuns.length > 0 && (
          <Table>
            <THead>
              <TR>
                <TH>Run</TH>
                <TH>Experiment</TH>
                <TH>Status</TH>
                <TH>Created</TH>
              </TR>
            </THead>
            <TBody>
              {recentRuns.map((run) => (
                <TR key={run.id}>
                  <TD className="font-medium">{run.run_name}</TD>
                  <TD className="text-muted-foreground">{run.experimentName}</TD>
                  <TD>
                    <ExperimentStatusBadge status={run.status} />
                  </TD>
                  <TD className="mono-value text-muted-foreground">
                    {new Date(run.created_at).toLocaleDateString()}
                  </TD>
                </TR>
              ))}
            </TBody>
          </Table>
        )}
      </Section>
    </div>
  );
}
