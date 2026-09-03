"use client";

import { cn } from "@/lib/cn";
import { Menu, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { BrandMark, Wordmark } from "./BrandMark";
import { NAV_SECTIONS } from "./nav-data";

export function MobileNav() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  return (
    <div className="flex items-center justify-between border-b border-border bg-card px-4 py-3 md:hidden">
      <Link href="/" className="flex items-center gap-2">
        <BrandMark size={22} />
        <span className="font-display text-[14px] font-semibold tracking-tight">AegisTrace</span>
      </Link>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label="Open navigation menu"
        className="flex h-8 w-8 items-center justify-center rounded-md text-foreground hover:bg-muted"
      >
        <Menu className="h-5 w-5" />
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex">
          <div className="absolute inset-0 bg-black/30" onClick={() => setOpen(false)} />
          <div className="relative flex h-full w-72 flex-col bg-card shadow-xl">
            <div className="flex items-center justify-between px-5 py-5">
              <div className="flex items-center gap-2.5">
                <BrandMark />
                <Wordmark />
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label="Close navigation menu"
                className="flex h-8 w-8 items-center justify-center rounded-md hover:bg-muted"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <nav className="flex-1 space-y-6 overflow-y-auto px-3 pb-6">
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
                          onClick={() => setOpen(false)}
                          className={cn(
                            "flex items-center gap-2.5 rounded-md py-2.5 px-3 text-[14px] transition-colors",
                            active ? "bg-primary-tint font-medium text-primary" : "text-foreground-secondary"
                          )}
                        >
                          <Icon className="h-4 w-4 shrink-0" strokeWidth={1.75} />
                          {item.label}
                        </Link>
                      );
                    })}
                  </div>
                </div>
              ))}
            </nav>
          </div>
        </div>
      )}
    </div>
  );
}
