"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// A small fixed palette tied to the app's gold/forensic identity rather
// than a generic rainbow — kept consistent across every chart in the app.
const SERIES_COLORS = ["#d6a13f", "#5cb583", "#e2645f", "#7c98b3", "#a487c2", "#c98b52"];

export function LineMetricChart({
  data,
  series,
  xKey,
  yLabel,
  height = 280,
}: {
  data: Record<string, number | string>[];
  series: string[];
  xKey: string;
  yLabel?: string;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey={xKey} tick={{ fontSize: 12 }} stroke="var(--muted-foreground)" />
        <YAxis
          tick={{ fontSize: 12 }}
          stroke="var(--muted-foreground)"
          label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fontSize: 12 } : undefined}
        />
        <Tooltip
          contentStyle={{
            background: "var(--card)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            fontSize: 12,
          }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {series.map((key, i) => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
            strokeWidth={2}
            dot={{ r: 3 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function BarMetricChart({
  data,
  series,
  xKey,
  yLabel,
  height = 280,
}: {
  data: Record<string, number | string>[];
  series: string[];
  xKey: string;
  yLabel?: string;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey={xKey} tick={{ fontSize: 12 }} stroke="var(--muted-foreground)" />
        <YAxis
          tick={{ fontSize: 12 }}
          stroke="var(--muted-foreground)"
          label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fontSize: 12 } : undefined}
        />
        <Tooltip
          contentStyle={{
            background: "var(--card)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            fontSize: 12,
          }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {series.map((key, i) => (
          <Bar key={key} dataKey={key} fill={SERIES_COLORS[i % SERIES_COLORS.length]} radius={[4, 4, 0, 0]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
