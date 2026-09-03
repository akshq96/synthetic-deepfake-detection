# Frontend — Deepfake Detection Dashboard

Next.js 16 (App Router) + TypeScript + Tailwind v4, client-rendered against
the FastAPI backend (`../backend/`) via TanStack Query. See the repo root
`README.md` for the full-stack quickstart and `../docs/api_reference.md` for
what each page talks to.

## Development

```bash
npm install
cp .env.local.example .env.local   # points NEXT_PUBLIC_API_BASE_URL at the backend
npm run dev
```

The backend must be running separately (`uvicorn backend.app.main:app` from
the repo root) for any page that fetches data to show anything other than
loading/error states.

## Testing

```bash
npm test          # vitest — component unit tests
npm run test:e2e  # Playwright — navigation always runs; the live detect
                   # happy-path test auto-skips if no backend is reachable
                   # at NEXT_PUBLIC_API_BASE_URL
npm run lint       # eslint
npx tsc --noEmit   # type-check
npm run build      # production build
```

## Structure

- `app/` — one route per required dashboard feature (see repo root README).
- `components/` — `ui/` (hand-rolled Tailwind primitives: Card, Button,
  Badge, Table, Tabs, Spinner — no component-library CLI dependency),
  `layout/` (sidebar nav), and feature components (UploadWidget,
  HeatmapOverlay, ConfidenceGauge, MetricChart, ComparisonTable,
  SuspiciousFrameStrip, ExperimentStatusBadge, ResultsLookupForm).
- `lib/api-client.ts` + `lib/types.ts` — a thin typed fetch wrapper and
  hand-kept-in-sync TypeScript mirrors of the backend's Pydantic schemas
  (no codegen — both sides are small and owned in this repo).
