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
    <svg viewBox="0 0 32 32" className="h-7 w-7 shrink-0" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="1" y="1" width="30" height="30" rx="7" stroke="var(--primary)" strokeWidth="1.5" />
      <path
        d="M9 16.5 14 21 23 10.5"
        stroke="var(--primary)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="23" cy="10.5" r="2.5" fill="var(--background)" stroke="var(--primary)" strokeWidth="1.5" />
    </svg>
  );
}

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-64 shrink-0 border-r border-border bg-card md:flex md:flex-col">
      <Link href="/" className="flex items-center gap-2.5 px-5 py-6">
        <BrandMark />
        <span className="leading-tight">
          <span className="block font-display text-[15px] font-semibold tracking-tight">
            Aegis<span className="text-primary">Trace</span>
          </span>
          <span className="label-mono block text-[10px] text-muted-foreground">
            Deepfake Forensics
          </span>
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
                      "relative flex items-center gap-2.5 rounded-md py-2 pl-3 pr-2.5 text-[13px] transition-colors",
                      active
                        ? "bg-primary/[0.08] font-medium text-primary"
                        : "text-foreground/75 hover:bg-muted hover:text-foreground"
                    )}
                  >
                    {active && (
                      <span className="absolute left-0 top-1/2 h-4 w-[2.5px] -translate-y-1/2 rounded-full bg-primary" />
                    )}
                    <Icon className="h-[15px] w-[15px] shrink-0" strokeWidth={1.75} />
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
