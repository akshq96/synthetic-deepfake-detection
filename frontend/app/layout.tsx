import type { Metadata } from "next";
import { Geist, Geist_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";
import { QueryProvider } from "@/lib/query-provider";
import { Sidebar } from "@/components/layout/Sidebar";
import { MobileNav } from "@/components/layout/MobileNav";
import { TopUtilityBar } from "@/components/layout/TopUtilityBar";

// Applies a stored theme choice before first paint, so ThemeToggle never
// causes a light->dark (or vice versa) flash on load. Runs for every page;
// safe as inline script since it only touches a data attribute.
const THEME_INIT_SCRIPT = `
(function () {
  try {
    var stored = window.localStorage.getItem("aegistrace-theme");
    if (stored === "light" || stored === "dark") {
      document.documentElement.dataset.theme = stored;
    }
  } catch (e) {}
})();
`;

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Deepfake Detection Dashboard",
  description: "Synthetic Data-Augmented Deepfake Detection — research dashboard",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${spaceGrotesk.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body className="min-h-full">
        <QueryProvider>
          <div className="flex min-h-screen flex-col md:flex-row">
            <Sidebar />
            <div className="flex min-w-0 flex-1 flex-col">
              <MobileNav />
              <TopUtilityBar />
              <main className="flex-1 px-4 py-5 sm:px-6 sm:py-6 md:px-8 md:py-8">{children}</main>
            </div>
          </div>
        </QueryProvider>
      </body>
    </html>
  );
}
