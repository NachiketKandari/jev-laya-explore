import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Laya Classifier Demo",
  description: "Visualise Laya customer-query classification results",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
