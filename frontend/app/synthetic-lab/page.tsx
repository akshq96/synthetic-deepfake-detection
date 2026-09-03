"use client";

import { ExperimentStatusBadge } from "@/components/ExperimentStatusBadge";
import { PageHeader } from "@/components/PageHeader";
import { ErrorState } from "@/components/QueryState";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { api } from "@/lib/api-client";
import { cn } from "@/lib/cn";
import type { SyntheticLabRunRequest } from "@/lib/types";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";

const BASE_CONFIGS = [
  { value: "ml/configs/cnn_baseline_debug.yaml", label: "CNN — debug (fast, CPU)" },
  { value: "ml/configs/cnn_baseline.yaml", label: "CNN — full run (GPU)" },
  { value: "ml/configs/vit_baseline_debug.yaml", label: "ViT — debug (fast, CPU)" },
  { value: "ml/configs/vit_baseline.yaml", label: "ViT — full run (GPU)" },
];

const TECHNIQUES = [
  { value: "blend_warp", label: "Blend / warp splicing" },
  { value: "freq_perturb", label: "Frequency-domain perturbation" },
  { value: "compression_artifact", label: "Compression artifact" },
  { value: "color_perturb", label: "Color / illumination mismatch" },
  { value: "autoencoder_swap", label: "Autoencoder reconstruction" },
];

export default function SyntheticLabPage() {
  const [experimentName, setExperimentName] = useState("synthetic_lab");
  const [runName, setRunName] = useState(() => `run_${Date.now()}`);
  const [baseConfig, setBaseConfig] = useState(BASE_CONFIGS[0].value);
  const [ratio, setRatio] = useState(0.3);
  const [techniques, setTechniques] = useState<string[]>(["blend_warp", "freq_perturb"]);
  const [manifestPath, setManifestPath] = useState("");
  const [launchedRunId, setLaunchedRunId] = useState<string | null>(null);

  const launchMutation = useMutation({
    mutationFn: (payload: SyntheticLabRunRequest) => api.runSyntheticLab(payload),
    onSuccess: (run) => setLaunchedRunId(run.id),
  });

  const statusQuery = useQuery({
    queryKey: ["run-status", launchedRunId],
    queryFn: () => api.getRunStatus(launchedRunId!),
    enabled: !!launchedRunId,
    refetchInterval: (query) => (query.state.data?.status === "running" ? 2000 : false),
  });

  function toggleTechnique(value: string) {
    setTechniques((prev) => (prev.includes(value) ? prev.filter((t) => t !== value) : [...prev, value]));
  }

  function handleLaunch() {
    launchMutation.mutate({
      experiment_name: experimentName,
      run_name: runName,
      base_config_path: baseConfig,
      synthetic_ratio: ratio,
      synthetic_techniques: ratio > 0 ? techniques : [],
      manifest_path: manifestPath || null,
    });
  }

  return (
    <div>
      <PageHeader
        title="Synthetic Data Lab"
        description="Configure a synthetic-augmentation ratio/technique mix and launch a training run — this is the baseline-vs-augmented condition the research question is answered from."
      />

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Run configuration</CardTitle>
            <CardDescription>
              Ratio 0 = baseline (no synthetic data). Techniques are ignored when ratio is 0.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Field label="Experiment name">
              <input
                className="input"
                value={experimentName}
                onChange={(e) => setExperimentName(e.target.value)}
              />
            </Field>
            <Field label="Run name">
              <input className="input" value={runName} onChange={(e) => setRunName(e.target.value)} />
            </Field>
            <Field label="Base config">
              <select className="input" value={baseConfig} onChange={(e) => setBaseConfig(e.target.value)}>
                {BASE_CONFIGS.map((c) => (
                  <option key={c.value} value={c.value}>
                    {c.label}
                  </option>
                ))}
              </select>
            </Field>
            <Field label={`Synthetic ratio: ${ratio.toFixed(2)}`}>
              <input
                type="range"
                min={0}
                max={0.9}
                step={0.05}
                value={ratio}
                onChange={(e) => setRatio(Number(e.target.value))}
                className="w-full accent-primary"
              />
            </Field>
            {ratio > 0 && (
              <Field label="Synthetic techniques">
                <div className="flex flex-wrap gap-2">
                  {TECHNIQUES.map((t) => (
                    <button
                      key={t.value}
                      type="button"
                      onClick={() => toggleTechnique(t.value)}
                      className={cn(
                        "rounded-full border px-3 py-1 text-xs transition-colors",
                        techniques.includes(t.value)
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-border text-muted-foreground hover:bg-muted"
                      )}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
              </Field>
            )}
            <Field label="Manifest path (optional override)">
              <input
                className="input"
                placeholder="e.g. data/manifests/ffpp_split.parquet"
                value={manifestPath}
                onChange={(e) => setManifestPath(e.target.value)}
              />
            </Field>

            <Button
              onClick={handleLaunch}
              disabled={launchMutation.isPending || (ratio > 0 && techniques.length === 0)}
              className="w-full"
            >
              Launch run
            </Button>
            {launchMutation.isError && <ErrorState error={launchMutation.error} />}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Run status</CardTitle>
            <CardDescription>Polls every 2 seconds while the subprocess is running.</CardDescription>
          </CardHeader>
          <CardContent>
            {!launchedRunId && (
              <p className="text-sm text-muted-foreground">Launch a run to see its status here.</p>
            )}
            {statusQuery.data && (
              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Status</span>
                  <ExperimentStatusBadge status={statusQuery.data.status} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Run ID</span>
                  <code className="text-xs">{statusQuery.data.id}</code>
                </div>
                {statusQuery.data.mlflow_run_id && (
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">MLflow run</span>
                    <code className="text-xs">{statusQuery.data.mlflow_run_id}</code>
                  </div>
                )}
                {statusQuery.data.error_message && (
                  <p className="rounded-md bg-danger/10 p-2 text-xs text-danger">
                    {statusQuery.data.error_message}
                  </p>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-medium text-muted-foreground">{label}</span>
      {children}
    </label>
  );
}
