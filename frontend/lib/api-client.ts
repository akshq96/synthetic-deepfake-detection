// Thin typed fetch wrapper around the FastAPI backend. Kept deliberately
// simple (no codegen) — see lib/types.ts for the hand-kept-in-sync response
// shapes.

import type {
  AblationResultPoint,
  CrossDatasetResult,
  ExperimentOut,
  MlflowRunSummary,
  ModelCompareOut,
  PredictionOut,
  ReportOut,
  RobustnessResult,
  RunOut,
  RunStatusOut,
  SyntheticLabRunRequest,
  UnseenManipulationResult,
} from "./types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(
      typeof detail === "string"
        ? detail
        : `API request failed with status ${status}`
    );
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let detail: unknown = null;
    try {
      const body = await response.json();
      detail = body?.detail ?? body;
    } catch {
      // response body wasn't JSON — leave detail null, status still conveys the failure
    }
    throw new ApiError(response.status, detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

function getJson<T>(path: string): Promise<T> {
  return request<T>(path, { method: "GET" });
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: JSON.stringify(body) });
}

function postForm<T>(path: string, form: FormData): Promise<T> {
  return request<T>(path, { method: "POST", body: form });
}

export function staticUrl(relativePath: string): string {
  return `${API_BASE_URL}/static/${relativePath}`;
}

export const api = {
  health: () => getJson<{ status: string }>("/api/health"),

  detectImage: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return postForm<PredictionOut>("/api/detect/image", form);
  },
  detectVideo: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return postForm<PredictionOut>("/api/detect/video", form);
  },
  getPrediction: (id: string) => getJson<PredictionOut>(`/api/detect/${id}`),

  listExperiments: () => getJson<ExperimentOut[]>("/api/experiments"),
  getExperiment: (id: string) => getJson<ExperimentOut>(`/api/experiments/${id}`),
  runSyntheticLab: (payload: SyntheticLabRunRequest) =>
    postJson<RunOut>("/api/experiments/synthetic-lab/run", payload),
  getRunStatus: (runId: string) =>
    getJson<RunStatusOut>(`/api/experiments/${runId}/status`),

  getBaselineVsSynthetic: (resultsDir: string) =>
    getJson<AblationResultPoint[]>(
      `/api/results/baseline-vs-synthetic?results_dir=${encodeURIComponent(resultsDir)}`
    ),
  getUnseenManipulation: (resultsDir: string) =>
    getJson<UnseenManipulationResult>(
      `/api/results/unseen-manipulation?results_dir=${encodeURIComponent(resultsDir)}`
    ),
  getCrossDataset: (resultsDir: string) =>
    getJson<CrossDatasetResult>(
      `/api/results/cross-dataset?results_dir=${encodeURIComponent(resultsDir)}`
    ),
  getRobustness: (resultsDir: string) =>
    getJson<RobustnessResult>(
      `/api/results/robustness?results_dir=${encodeURIComponent(resultsDir)}`
    ),
  getAblation: (resultsDir: string) =>
    getJson<AblationResultPoint[]>(
      `/api/results/ablation?results_dir=${encodeURIComponent(resultsDir)}`
    ),

  compareModels: (experimentName: string) =>
    getJson<ModelCompareOut>(
      `/api/models/compare?experiment_name=${encodeURIComponent(experimentName)}`
    ),
  listMlflowRuns: (experimentName?: string) =>
    getJson<MlflowRunSummary[]>(
      `/api/mlflow/runs${experimentName ? `?experiment_name=${encodeURIComponent(experimentName)}` : ""}`
    ),

  generateReport: (predictionId: string) =>
    postJson<ReportOut>("/api/reports/generate", { prediction_id: predictionId }),
  listReports: () => getJson<ReportOut[]>("/api/reports"),
  getReport: (id: string) => getJson<ReportOut>(`/api/reports/${id}`),
  reportExportUrl: (reportId: string) => `${API_BASE_URL}/api/reports/${reportId}/export`,
};
