import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Axiom-I | Image Forensics',
  description:
    'Deepfake detection through physics-based image forensics. Specular residual analysis, frequency domain inspection, wavelet decomposition, and ViT classification.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>{children}</body>
    </html>
  );
}
