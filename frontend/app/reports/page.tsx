"use client";

import { PageHeader } from "@/components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "@/components/QueryState";
import { Card, CardContent } from "@/components/ui/Card";
import { api } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";
import { FileText } from "lucide-react";
import Link from "next/link";

export default function ReportsPage() {
  const query = useQuery({ queryKey: ["reports"], queryFn: api.listReports });

  return (
    <div>
      <PageHeader
        title="Forensic Reports"
        description="Generated forensic PDF reports — created from the Detection page after running a prediction."
      />

      {query.isLoading && <LoadingState />}
      {query.isError && <ErrorState error={query.error} />}
      {query.isSuccess && query.data.length === 0 && (
        <EmptyState label="No reports yet — generate one from the Detection page." />
      )}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {query.data?.map((report) => (
          <Link key={report.id} href={`/reports/${report.id}`}>
            <Card className="h-full transition-colors hover:border-primary/50">
              <CardContent className="flex items-center gap-3 pt-5">
                <FileText className="h-5 w-5 shrink-0 text-primary" />
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium">Report {report.id.slice(0, 8)}</div>
                  <div className="text-xs text-muted-foreground">
                    {new Date(report.created_at).toLocaleString()}
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
