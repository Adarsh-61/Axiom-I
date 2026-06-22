# Axiom-I Frontend Web Client Guide

Welcome to the frontend presentation layer documentation for Axiom-I. This is a Next.js web application engineered to provide a visual workspace for digital forensics analysts. The interface facilitates media ingestion, coordinates backend API requests, maps dynamic execution pathways, and renders step-by-step mathematical proofs.

Official project links:
* GitHub Repository: https://github.com/Adarsh-61/Axiom-I
* Frontend: https://axiom-i.vercel.app/
* Alternative Domain: https://axiom.dpdns.org/
* Backend Space: https://huggingface.co/spaces/Adarsh-61/axiom-backend

---

## 1. Technology Stack

* **Framework**: Next.js 14 (App Router)
* **Language**: TypeScript
* **Styling**: Vanilla CSS utilizing CSS variables for theme and responsive grid layouts.
* **Mathematical Rendering**: KaTeX (via custom Tex rendering bindings)

---

## 2. Folder Structure

```
frontend/
  src/
    app/
      page.tsx             - Main forensic analysis dashboard
      layout.tsx           - HTML shell containing viewport and SEO meta tags
      globals.css          - Comprehensive visual styles and layout grids
      helpers.ts           - API client utilities, schemas, and pipeline mappings
      Tex.tsx              - Safe KaTeX typesetting component wrapper
      math/
        page.tsx           - Full mathematical formula derivation page
  package.json             - Project dependency catalog
```

---

## 3. Installation and Local Setup

Ensure Node.js 20+ and npm are installed on your machine.

### Installation Steps

1. Navigate to the frontend directory:
   ```bash
   cd Coding/frontend
   ```
2. Install npm package dependencies:
   ```bash
   npm install
   ```
3. Configure environment variables:
   Create or edit `Coding/frontend/.env.local` to define the API target URL.
   Ensure both local development and hosted production endpoints are clear:
   ```env
   # Local Development
   NEXT_PUBLIC_API_BASE=http://localhost:8000

   # Hosted Backend Production
   NEXT_PUBLIC_API_BASE=https://huggingface.co/spaces/Adarsh-61/axiom-backend
   ```

---

## 4. Local Development Commands

Run these commands from `Coding/frontend/`:
* `npm run dev`: Starts a local development server on http://localhost:3000.
* `npm run build`: Compiles production-optimized code bundles.
* `npm run start`: Starts the compiled production server.
* `npm run lint`: Triggers the TypeScript compiler syntax check (`tsc --noEmit`).

---

## 5. UI Features and Page Routes

### 5.1 Main Forensic Workspace (`/`)
* **Media Upload**: Supports drag-and-drop or manual selection for image and video formats (up to 50 MB).
* **Result Panel**: Displays the verdict, confidence percentage, execution mode (e.g., Full Physics, Fallback, Video Physics, Video Fallback), face counts, and calibration telemetry.
* **Signal Anomaly Tracks**: Renders horizontal bar meters colored by severity (Green for safe, Amber for moderate, Red for high anomaly) representing individual signal contributions.
* **Execution Flow Graph**: Maps each analytical step to its respective backend file name, detailing the mathematical formulas and intermediate visualizations (such as Bilateral depth maps, Multi-Scale Retinex textures, and Specular residual heatmaps).

### 5.2 Mathematics Proofs Page (`/math`)
Provides a complete mathematical breakdown of the current analysis. It loads the full JSON results from the session state, rendering equations and computed values using KaTeX.
* **Image Physics**: Displays surface normal vector equations, Multi-Scale Retinex logarithmic calculations, 9-term Spherical Harmonics coefficients, Lambertian rendering equations, and Noisy-OR Bayesian products.
* **Video Physics**: Details frame sampling rates, rPPG POS projection matrices, optical flow divergence variance, temporal wavelet variances, and video signal fusion weights.

---

## 6. Mathematical Sign Typo Verification

All formulas displayed on the frontend have been validated against the backend ML algorithms. For example:
* **Specular correlation**: Uses Normalized Cross-Correlation (NCC). The sigmoid threshold mapping function is:
  `sigma(15 * (NCC - 0.30))`
* **FFT High-Frequency Energy Ratio (HFER)**: Uses shifted logarithmic sigmoid scaling:
  `1 / (1 + exp(-3.0 * (log10(HFER) + 4.5)))`
* **Patch consistency (PRNU)**: Evaluates noise standard deviation over local SNR matrices:
  `1 / (1 + exp(-20 * (CV - 0.25)))`

---

## 7. Troubleshooting and FAQ

### Why does my image analysis switch to Fallback mode?
The system utilizes the MTCNN neural network to isolate faces. If the upload does not contain a face or the detector confidence is below 90%, the system switches to Fallback mode, running analysis on the entire image canvas using a subset of signals (Frequency, Wavelet, and ViT).

### Why does the video health check show degraded status?
On CPU-only hosting environments, the pre-trained Vision Transformer model is kept inactive by default (`AXIOM_ALLOW_MODEL_DOWNLOAD=false`) to avoid long container start times. In this state, the health check returns "degraded", and fallback models execute successfully.

---

## 8. License

This project is licensed under the MIT License. See the public [LICENSE](https://github.com/Adarsh-61/Axiom-I/blob/main/LICENSE) file for details.
