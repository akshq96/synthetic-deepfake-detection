"use client";

import { ComparisonTable, type ComparisonColumn } from "@/components/ComparisonTable";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "@/components/QueryState";
import { ResultsLookupForm } from "@/components/ResultsLookupForm";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
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
        title="Generalization"
        description="Unseen-manipulation (leave-one-manipulation-out) and cross-dataset evaluation — the results that most directly test generalization."
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

  return (
    <div>
      <ResultsLookupForm
        label="Results directory (relative to artifacts/)"
        placeholder="experiments/unseen_manipulation/2026-01-01T00-00-00"
        onSubmit={setResultsDir}
      />
      {query.isLoading && <LoadingState />}
      {query.isError && <ErrorState error={query.error} />}
      {query.isSuccess && rows.length === 0 && <EmptyState label="No manipulation types found in this results file." />}
      {rows.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Held-out-manipulation accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <ComparisonTable rows={rows} columns={UNSEEN_COLUMNS} defaultSortKey="manipulationType" />
          </CardContent>
        </Card>
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

  return (
    <div>
      <ResultsLookupForm
        label="Results directory (relative to artifacts/)"
        placeholder="experiments/cross_dataset/2026-01-01T00-00-00"
        onSubmit={setResultsDir}
      />
      {query.isLoading && <LoadingState />}
      {query.isError && <ErrorState error={query.error} />}
      {query.data && (
        <Card>
          <CardHeader>
            <CardTitle>
              {query.data.train_dataset} → {query.data.eval_dataset}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
              <Stat
                label="In-distribution accuracy"
                value={query.data.in_distribution_metrics?.accuracy}
              />
              <Stat label="Cross-dataset accuracy" value={query.data.cross_dataset_metrics?.accuracy} />
              <Stat label="Generalization gap" value={query.data.generalization_gap_accuracy} highlight />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function Stat({ label, value, highlight }: { label: string; value: number | null | undefined; highlight?: boolean }) {
  return (
    <div className="rounded-lg bg-muted p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={`mt-1 text-lg font-semibold ${highlight ? "text-primary" : ""}`}>
        {value == null ? "—" : value.toFixed(3)}
      </div>
      {highlight && value != null && (
        <Badge variant={value > 0.1 ? "warning" : "success"} className="mt-1">
          {value > 0.1 ? "notable drop" : "small drop"}
        </Badge>
      )}
    </div>
  );
}
