"use client";

import { ComparisonTable, type ComparisonColumn } from "@/components/ComparisonTable";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "@/components/QueryState";
import { ResultsLookupForm } from "@/components/ResultsLookupForm";
import { Section } from "@/components/ui/Section";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import { api } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

interface UnseenRow {
  id: string;
  manipulationType: string;
  status: string;
  accuracy: number | null;
  rocAuc: number | null;
  nSamples: number | null;
}

const UNSEEN_COLUMNS: ComparisonColumn<UnseenRow>[] = [
  { key: "manipulationType", label: "Held-out type" },
  { key: "status", label: "Status" },
  { key: "accuracy", label: "Accuracy", format: (v) => (v == null ? "—" : Number(v).toFixed(3)) },
  { key: "rocAuc", label: "ROC-AUC", format: (v) => (v == null ? "—" : Number(v).toFixed(3)) },
  { key: "nSamples", label: "N eval samples", format: (v) => (v == null ? "—" : String(v)) },
];

export default function GeneralizationPage() {
  return (
    <div>
      <PageHeader
        eyebrow="Generalization"
        title="Known vs. Unseen Manipulation"
        description="Unseen-manipulation (leave-one-manipulation-out) and cross-dataset evaluation — the results that most directly test generalization to patterns the model never saw during training."
      />
      <Tabs defaultValue="unseen">
        <TabsList>
          <TabsTrigger value="unseen">Unseen manipulation</TabsTrigger>
          <TabsTrigger value="cross-dataset">Cross-dataset</TabsTrigger>
        </TabsList>
        <TabsContent value="unseen">
          <UnseenManipulationTab />
        </TabsContent>
        <TabsContent value="cross-dataset">
          <CrossDatasetTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}

function UnseenManipulationTab() {
  const [resultsDir, setResultsDir] = useState<string | null>(null);
  const query = useQuery({
    queryKey: ["unseen-manipulation", resultsDir],
    queryFn: () => api.getUnseenManipulation(resultsDir!),
    enabled: !!resultsDir,
  });

  const rows: UnseenRow[] = query.data
    ? Object.entries(query.data).map(([type, result]) => ({
        id: type,
        manipulationType: type,
        status: result.skipped ? "skipped" : "evaluated",
        accuracy: result.skipped ? null : result.eval_metrics.accuracy,
        rocAuc: result.skipped ? null : result.eval_metrics.roc_auc,
        nSamples: result.skipped ? null : result.n_eval_samples,
      }))
    : [];
  const evaluated = rows.filter((r) => r.accuracy != null);

  return (
    <div className="pt-5">
      <ResultsLookupForm
        label="Results directory (relative to artifacts/)"
        placeholder="experiments/unseen_manipulation/2026-01-01T00-00-00"
        onSubmit={setResultsDir}
      />
      {query.isLoading && <LoadingState />}
      {query.isError && (
        <ErrorState
          error={query.error}
          title="Unable to load unseen-manipulation results"
          onRetry={() => query.refetch()}
        />
      )}
      {query.isSuccess && rows.length === 0 && (
        <EmptyState
          title="No manipulation types found"
          description="Run ml.evaluation.experiments.unseen_manipulation, then load its results directory here."
        />
      )}
      {rows.length > 0 && (
        <div>
          {evaluated.length > 0 && (
            <Section
              title="Accuracy when each type is held out entirely"
              description="Each bar: accuracy on a manipulation type the model never saw during training. Skipped types had no held-out evaluation samples at this manifest's scale."
            >
              <div className="space-y-3">
                {evaluated.map((row) => (
                  <div key={row.id} className="flex items-center gap-3">
                    <span className="w-32 shrink-0 truncate text-[12.5px] text-foreground-secondary">
                      {row.manipulationType}
                    </span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-primary"
                        style={{ width: `${Math.min(100, (row.accuracy ?? 0) * 100)}%` }}
                      />
                    </div>
                    <span className="mono-value w-14 shrink-0 text-right text-[12.5px]">
                      {((row.accuracy ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </Section>
          )}
          <Section title="Full results">
            <ComparisonTable rows={rows} columns={UNSEEN_COLUMNS} defaultSortKey="manipulationType" />
          </Section>
        </div>
      )}
    </div>
  );
}

function CrossDatasetTab() {
  const [resultsDir, setResultsDir] = useState<string | null>(null);
  const query = useQuery({
    queryKey: ["cross-dataset", resultsDir],
    queryFn: () => api.getCrossDataset(resultsDir!),
    enabled: !!resultsDir,
  });

  const inDist = query.data?.in_distribution_metrics?.accuracy;
  const cross = query.data?.cross_dataset_metrics?.accuracy;
  const gap = query.data?.generalization_gap_accuracy;

  return (
    <div className="pt-5">
      <ResultsLookupForm
        label="Results directory (relative to artifacts/)"
        placeholder="experiments/cross_dataset/2026-01-01T00-00-00"
        onSubmit={setResultsDir}
      />
      {query.isLoading && <LoadingState />}
      {query.isError && (
        <ErrorState error={query.error} title="Unable to load cross-dataset results" onRetry={() => query.refetch()} />
      )}
      {query.data && (
        <Section title={`${query.data.train_dataset} → ${query.data.eval_dataset}`}>
          <div className="space-y-4">
            <BarRow label="Known dataset (in-distribution)" value={inDist} />
            <BarRow label="Unseen dataset (cross-dataset)" value={cross} />
          </div>
          {gap != null && (
            <div className="mt-5 flex items-center gap-2 border-t border-border pt-4 text-[13px]">
              <span className="text-muted-foreground">Generalization gap</span>
              <span className={`mono-value font-medium ${gap > 0.1 ? "text-warning" : "text-success"}`}>
                {(gap * 100).toFixed(1)} pts
              </span>
              <span className="text-muted-foreground">{gap > 0.1 ? "— a notable drop" : "— a small drop"}</span>
            </div>
          )}
        </Section>
      )}
    </div>
  );
}

function BarRow({ label, value }: { label: string; value: number | null | undefined }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-56 shrink-0 text-[12.5px] text-foreground-secondary">{label}</span>
      <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-muted">
        <div className="h-full rounded-full bg-primary" style={{ width: `${Math.min(100, (value ?? 0) * 100)}%` }} />
      </div>
      <span className="mono-value w-14 shrink-0 text-right text-[12.5px]">
        {value == null ? "—" : `${(value * 100).toFixed(1)}%`}
      </span>
    </div>
  );
}
