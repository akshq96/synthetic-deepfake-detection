"use client";

import { cn } from "@/lib/cn";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  FlaskConical,
  GitCompareArrows,
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

function BrandMark() {
  return (
    <div className="relative flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-primary-2 shadow-[0_0_20px_-4px_var(--primary)]">
      <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none">
        <path
          d="M4 9.5 8 13.5 16 5.5"
          stroke="white"
          strokeWidth="2.25"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-64 shrink-0 border-r border-border bg-card/60 backdrop-blur-xl md:flex md:flex-col">
      <Link href="/" className="flex items-center gap-2.5 px-5 py-6">
        <BrandMark />
        <span className="leading-tight">
          <span className="font-display gradient-text block text-[15px] font-semibold tracking-tight">
            AegisTrace
          </span>
          <span className="block text-[10.5px] text-muted-foreground">Deepfake Forensics</span>
        </span>
      </Link>

      <nav aria-label="Primary" className="flex-1 space-y-7 overflow-y-auto px-3 pb-6">
        {NAV_SECTIONS.map((section) => (
          <div key={section.title}>
            <div className="label-mono px-3 pb-2 text-[10px] text-muted-foreground/70">
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
                      "relative flex items-center gap-2.5 rounded-lg py-2 px-3 text-[13px] transition-all",
                      active
                        ? "bg-gradient-to-r from-primary/15 to-primary-2/10 font-medium text-foreground shadow-[inset_0_0_0_1px_color-mix(in_srgb,var(--primary)_35%,transparent)]"
                        : "text-foreground/70 hover:bg-muted hover:text-foreground"
                    )}
                  >
                    <Icon
                      className={cn("h-[15px] w-[15px] shrink-0", active && "text-primary")}
                      strokeWidth={1.75}
                    />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="label-mono border-t border-border px-5 py-3 text-[10px] text-muted-foreground/60">
        Research Build
      </div>
    </aside>
  );
}
