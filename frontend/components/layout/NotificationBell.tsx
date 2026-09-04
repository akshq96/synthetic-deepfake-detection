"use client";

import { Bell } from "lucide-react";
import { useState } from "react";

// Static placeholder for Milestone A — a real, session-scoped notification
// feed (detect completions, report generation) lands in Milestone E. Kept
// as a genuinely empty state rather than a fake badge count in the meantime.
export function NotificationBell() {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label="Notifications"
        title="Notifications"
        className="flex h-8 w-8 items-center justify-center rounded-md border border-border text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      >
        <Bell className="h-4 w-4" />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-10 z-50 w-64 rounded-lg border border-border bg-card p-4 shadow-lg">
            <div className="meta-label mb-1 text-[10px]">Notifications</div>
            <p className="text-[13px] text-muted-foreground">No notifications yet.</p>
          </div>
        </>
      )}
    </div>
  );
}
