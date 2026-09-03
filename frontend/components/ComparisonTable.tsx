"use client";

import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/Table";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import { useMemo, useState } from "react";

export interface ComparisonColumn<Row> {
  key: keyof Row & string;
  label: string;
  format?: (value: Row[keyof Row]) => React.ReactNode;
}

export function ComparisonTable<Row extends { id: string }>({
  rows,
  columns,
  defaultSortKey,
}: {
  rows: Row[];
  columns: ComparisonColumn<Row>[];
  defaultSortKey?: keyof Row & string;
}) {
  const [sortKey, setSortKey] = useState<string | null>(defaultSortKey ?? null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const sortedRows = useMemo(() => {
    if (!sortKey) return rows;
    const copy = [...rows];
    copy.sort((a, b) => {
      const av = a[sortKey as keyof Row];
      const bv = b[sortKey as keyof Row];
      if (typeof av === "number" && typeof bv === "number") {
        return sortDir === "asc" ? av - bv : bv - av;
      }
      return sortDir === "asc"
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
    return copy;
  }, [rows, sortKey, sortDir]);

  function toggleSort(key: string) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  return (
    <Table>
      <THead>
        <TR>
          {columns.map((col) => (
            <TH key={col.key}>
              <button
                type="button"
                onClick={() => toggleSort(col.key)}
                className="inline-flex items-center gap-1 hover:text-foreground"
              >
                {col.label}
                {sortKey === col.key ? (
                  sortDir === "asc" ? (
                    <ArrowUp className="h-3 w-3" />
                  ) : (
                    <ArrowDown className="h-3 w-3" />
                  )
                ) : (
                  <ArrowUpDown className="h-3 w-3 opacity-40" />
                )}
              </button>
            </TH>
          ))}
        </TR>
      </THead>
      <TBody>
        {sortedRows.map((row) => (
          <TR key={row.id}>
            {columns.map((col) => (
              <TD key={col.key}>
                {col.format ? col.format(row[col.key]) : String(row[col.key] ?? "—")}
              </TD>
            ))}
          </TR>
        ))}
        {sortedRows.length === 0 && (
          <tr>
            <td colSpan={columns.length} className="px-3 py-6 text-center text-muted-foreground">
              No data yet
            </td>
          </tr>
        )}
      </TBody>
    </Table>
  );
}
