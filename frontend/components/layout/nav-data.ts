import {
  FlaskConical,
  GitCompareArrows,
  ScanSearch,
  ShieldAlert,
  Sparkles,
  FileText,
  History,
  Video,
  Bookmark,
} from "lucide-react";

export const NAV_SECTIONS: {
  title: string;
  items: { href: string; label: string; icon: React.ElementType; isNew?: boolean }[];
}[] = [
  {
    title: "Detect",
    items: [
      { href: "/detect", label: "Image / Video Detection", icon: ScanSearch },
      { href: "/live", label: "Live Camera Detection", icon: Video, isNew: true },
    ],
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
      { href: "/saved-analyses", label: "Saved Analyses", icon: Bookmark },
    ],
  },
];
