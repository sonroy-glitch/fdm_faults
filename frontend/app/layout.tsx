import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FDM Fault Classifier",
  description:
    "Upload a photo of a 3D print and get the predicted FDM fault class with per-class confidence.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="text-slate-100 antialiased">{children}</body>
    </html>
  );
}
