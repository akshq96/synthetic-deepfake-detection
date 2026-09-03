"use client";

import { cn } from "@/lib/cn";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  FlaskConical,
  GitCompareArrows,
  LayoutDashboard,
  ScanSearch,
  ShieldAlert,
  Sparkles,
  FileText,
  History,
} from "lucide-react";

const NAV_SECTIONS: { title: string; items: { href: string; label: string; icon: React.ElementType }[] }[] = [
  {
    title: "Detect",
    items: [{ href: "/detect", label: "Image / Video Detection", icon: ScanSearch }],
  },
  {
    title: "Research",
    items: [
      { href: "/synthetic-lab", label: "Synthetic Data Lab", icon: Sparkles },
      { href: "/compare/cnn-vit", label: "CNN vs ViT", icon: GitCompareArrows },
      { href: "/results/baseline-vs-augmented", label: "Baseline vs Augmented", icon: FlaskConical },
      { href: "/results/generalization", label: "Generalization", icon: GitCompareArrows },
      { href: "/results/robustness", label: "Robustness", icon: ShieldAlert },
    ],
  },
  {
    title: "Records",
    items: [
      { href: "/experiments", label: "Experiment History", icon: History },
      { href: "/reports", label: "Forensic Reports", icon: FileText },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-64 shrink-0 border-r border-border bg-card md:flex md:flex-col">
      <Link href="/" className="flex items-center gap-2 px-5 py-5">
        <LayoutDashboard className="h-5 w-5 text-primary" />
        <span className="text-sm font-semibold leading-tight">
          Deepfake Detection
          <span className="block text-xs font-normal text-muted-foreground">Research Dashboard</span>
        </span>
      </Link>

      <nav className="flex-1 space-y-6 overflow-y-auto px-3 pb-6">
        {NAV_SECTIONS.map((section) => (
          <div key={section.title}>
            <div className="px-2 pb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              {section.title}
            </div>
            <div className="space-y-0.5">
              {section.items.map((item) => {
                const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition-colors",
                      active
                        ? "bg-primary/10 font-medium text-primary"
                        : "text-foreground/80 hover:bg-muted hover:text-foreground"
                    )}
                  >
                    <Icon className="h-4 w-4 shrink-0" />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>
    </aside>
  );
}
