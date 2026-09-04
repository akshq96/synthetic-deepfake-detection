import { NotificationBell } from "./NotificationBell";
import { ThemeToggle } from "./ThemeToggle";

// A slim persistent strip above every page's content, holding controls that
// apply app-wide (theme, notifications) — as opposed to page-specific
// actions like the Detect page's model selector, which stay in that page's
// own PageHeader instead of living here.
export function TopUtilityBar() {
  return (
    <div className="flex items-center justify-end gap-2 border-b border-border px-4 py-2.5 sm:px-6 md:px-8">
      <NotificationBell />
      <ThemeToggle />
    </div>
  );
}
