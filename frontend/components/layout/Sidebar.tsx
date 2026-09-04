"use client";

import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/cn";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { BrandMark, Wordmark } from "./BrandMark";
import { NAV_SECTIONS } from "./nav-data";
import { ProfileBlock } from "./ProfileBlock";

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-60 shrink-0 border-r border-border bg-card md:flex md:flex-col">
      <Link href="/" className="flex items-center gap-2.5 px-5 py-6">
        <BrandMark />
        <Wordmark />
      </Link>

      <nav aria-label="Primary" className="flex-1 space-y-6 overflow-y-auto px-3 pb-6">
        {NAV_SECTIONS.map((section) => (
          <div key={section.title}>
            <div className="meta-label px-3 pb-1.5 text-[10px]">{section.title}</div>
            <div className="space-y-px">
              {section.items.map((item) => {
                const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center gap-2.5 border-l-2 py-1.5 pl-3 pr-2.5 text-[13px] transition-colors",
                      active
                        ? "border-primary font-medium text-foreground"
                        : "border-transparent text-foreground-secondary hover:border-border-strong hover:text-foreground"
                    )}
                  >
                    <Icon className={cn("h-4 w-4 shrink-0", active ? "text-primary" : "text-muted-foreground")} strokeWidth={1.75} />
                    <span className="truncate">{item.label}</span>
                    {item.isNew && (
                      <Badge dot={false} className="ml-auto shrink-0 px-1.5 py-0 text-[9px]">
                        NEW
                      </Badge>
                    )}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <ProfileBlock />
    </aside>
  );
}
