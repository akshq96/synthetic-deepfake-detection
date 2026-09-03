// Mirrors backend/app/schemas/*.py — hand-kept in sync rather than codegen'd,
// since both sides are small and owned in this same repo.

export type PredictionLabel = "real" | "fake" | "abstain";

export interface SuspiciousFrameOut {
  track_id: number;
  frame_index: number;
  timestamp: number;
  fake_probability: number;
  heatmap_path: string | null;
}

export interface PredictionOut {
  id: string;
  input_type: "image" | "video";
  model_name: string;
  label: PredictionLabel;
  confidence: number;
  fake_probability: number;
  abstained: boolean;
  heatmap_path: string | null;
  created_at: string;
  suspicious_frames: SuspiciousFrameOut[];
}

export interface RunOut {
  id: string;
  experiment_id: string;
  run_name: string;
  mlflow_run_id: string | null;
  status: "pending" | "running" | "completed" | "failed";
  pid: number | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface ExperimentOut {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  runs: RunOut[];
}

export interface RunStatusOut {
  id: string;
  status: "pending" | "running" | "completed" | "failed";
  pid: number | null;
  error_message: string | null;
  mlflow_run_id: string | null;
}

export interface SyntheticLabRunRequest {
  experiment_name: string;
  run_name: string;
  base_config_path?: string;
  model_name?: string | null;
  synthetic_ratio: number;
  synthetic_techniques: string[];
  manifest_path?: string | null;
  epochs?: number | null;
}

export interface ReportOut {
  id: string;
  prediction_id: string;
  pdf_path: string;
  created_at: string;
}

export interface MlflowRunSummary {
  run_id: string;
  run_name: string | null;
  experiment_id: string;
  status: string;
  start_time: number;
  params: Record<string, string>;
  metrics: Record<string, number>;
}

export interface ModelCompareEntry {
  run_id: string;
  run_name: string | null;
  params: Record<string, string>;
  metrics: Record<string, number>;
}

export interface ModelCompareOut {
  experiment_name: string;
  models: Record<string, ModelCompareEntry>;
}

// Result-file shapes read back from ml/evaluation/experiments/*.py and
// ml/evaluation/ablation.py — see backend/app/api/results.py.
export interface EvalMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  n_samples: number;
}

export interface AblationResultPoint {
  point: Record<string, number | string>;
  run_summary: { run_id: string; mlflow_run_id: string; checkpoint_path: string };
  eval_metrics: EvalMetrics;
}

export interface UnseenManipulationResult {
  [manipulationType: string]:
    | { skipped: true; reason: string }
    | {
        skipped: false;
        held_out_type: string;
        n_eval_samples: number;
        eval_metrics: EvalMetrics;
        train_summary: { run_id: string; mlflow_run_id: string; checkpoint_path: string };
      };
}

export interface CrossDatasetResult {
  train_dataset: string;
  eval_dataset: string;
  train_summary: { run_id: string; mlflow_run_id: string; checkpoint_path: string };
  in_distribution_metrics: EvalMetrics | null;
  cross_dataset_metrics: EvalMetrics | null;
  generalization_gap_accuracy: number | null;
}

export interface RobustnessResult {
  checkpoint_path: string;
  compression_in_training_mix: boolean;
  baseline_metrics: EvalMetrics;
  perturbations: Record<string, { severity: number; metrics: EvalMetrics }[]>;
}
