export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';
export const FEEDBACK_CLIENT_ID_KEY = 'axiom_feedback_client_id';
export const ANALYZE_TIMEOUT_MS = 120_000;
export const FEEDBACK_TIMEOUT_MS = 30_000;
export const METRICS_TIMEOUT_MS = 30_000;

export interface SignalBreakdown {
  specular: number; frequency: number; topology: number;
  patch_consistency: number; wavelet_score: number; vit_score: number;
  physics_ensemble: number; raw_fusion: number; calibrated: number;
}
export interface FaceResult {
  bbox: number[]; confidence: number; verdict: string; score: number;
  signal_breakdown?: SignalBreakdown;
}
export interface VisualizationStep { step: number; label: string; data: string; }
export interface AnalysisResponse {
  verdict: string; confidence: number; faces_detected: number;
  faces: FaceResult[]; steps: VisualizationStep[];
  full_image_score?: number; analysis_mode?: string;
  fallback_breakdown?: Record<string, number>;
  feature_vector?: number[];
  calibration_breakdown?: Record<string, number>;
  quality_metrics?: Record<string, number>;
  process_inputs?: { image_shape?: number[]; analysis_mode?: string; components?: string[]; feature_names?: string[]; };
  decision_factors?: Record<string, number>;
  explanation?: string[]; error?: string;
  video_id?: string;
}
export interface FeedbackDiagnosticsResponse {
  confusion_matrix: { TP: number; TN: number; FP: number; FN: number; total: number; };
  calibration_metrics: Record<string, unknown>;
  calibration_history: Array<Record<string, unknown>>;
  feedback_summary: { total_feedback_records: number; training_eligible_records: number; training_excluded_records: number; training_exclusion_reasons: Record<string, number>; trust_summary?: Record<string, unknown>; };
}
export interface FeedbackSubmitResponse {
  status: string; message: string;
  confusion_matrix: { TP: number; TN: number; FP: number; FN: number; total?: number; };
  training_eligible: boolean; training_exclusion_reason?: string | null;
  user_trust_score?: number; user_sample_weight?: number;
}
export type ErrorPayload = { detail?: string; error?: string; message?: string; };

export const withTimeout = async (url: string, init: RequestInit, timeoutMs: number): Promise<Response> => {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  try { return await fetch(url, { ...init, signal: controller.signal }); }
  finally { window.clearTimeout(timeoutId); }
};
export const parseJsonSafe = async <T,>(response: Response): Promise<T | null> => {
  const text = await response.text();
  if (!text) return null;
  try { return JSON.parse(text) as T; } catch { return null; }
};
export const extractErrorMessage = (payload: ErrorPayload | null, fallback: string): string => {
  if (!payload) return fallback;
  if (typeof payload.detail === 'string' && payload.detail.trim()) return payload.detail;
  if (typeof payload.error === 'string' && payload.error.trim()) return payload.error;
  if (typeof payload.message === 'string' && payload.message.trim()) return payload.message;
  return fallback;
};
export const getOrCreateClientId = (): string => {
  if (typeof window === 'undefined') return 'anonymous';
  const existing = window.localStorage.getItem(FEEDBACK_CLIENT_ID_KEY);
  if (existing && existing.trim().length > 0) return existing;
  const created = typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID() : `client_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
  window.localStorage.setItem(FEEDBACK_CLIENT_ID_KEY, created);
  return created;
};
export const toPercent = (v: number | undefined): string => typeof v !== 'number' || Number.isNaN(v) ? '-' : `${(v * 100).toFixed(1)}%`;
export const toFixed = (v: number | undefined, d = 4): string => typeof v !== 'number' || Number.isNaN(v) ? '-' : v.toFixed(d);
export const toFileSize = (bytes: number): string => {
  if (!Number.isFinite(bytes) || bytes <= 0) return '-';
  return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
};

export interface PipelineStepDef {
  file: string; title: string; desc: string;
  imageStepIndex?: number; outputKey?: string; outputLabel?: string;
  latex?: string[];
  formulaComputed?: string;
}

const fmt = (v: number | undefined) => typeof v === 'number' ? v.toFixed(4) : '?';

export function getFullPhysicsSteps(r: AnalysisResponse): PipelineStepDef[] {
  const df = r.decision_factors || {};
  const sb = r.faces?.[0]?.signal_breakdown;
  const cb = r.calibration_breakdown || {};
  return [
    { file: 'face_detector.py', title: 'Step 1: Face Detection (MTCNN)', desc: `The input image is passed to the MTCNN neural network. It scans the image for face regions with at least 90% confidence and a minimum size of 64x64 pixels. The largest detected face is selected ${r.faces?.[0] ? `(Bounding Box: [${r.faces[0].bbox.join(', ')}], Confidence: ${(r.faces[0].confidence * 100).toFixed(2)}%)` : ''} and cropped with 10% padding on each side.`, imageStepIndex: 2, outputLabel: 'Faces detected', outputKey: 'faces_detected' },
    { file: 'face_alignment.py', title: 'Step 2: Surface Normal Estimation', desc: 'The face crop is converted to grayscale and smoothed with a bilateral filter (d=9, sigmaColor=75, sigmaSpace=75). This creates a pseudo depth map. A radial prior is blended in. Then Sobel gradients are computed and normals are built, normalized to unit length.', imageStepIndex: 3,
      latex: [
        '\\text{depth} = 0.75 \\cdot \\text{smoothed} + 0.25 \\cdot (1 - r^2)',
        '\\mathbf{N} = \\frac{[-2 \\cdot d_x,\\; -2 \\cdot d_y,\\; 1]}{\\|[-2 \\cdot d_x,\\; -2 \\cdot d_y,\\; 1]\\|}',
      ] },
    { file: 'retinex.py', title: 'Step 3: Multi-Scale Retinex (MSR)', desc: 'The Retinex theory separates an image into reflectance (texture) and illumination. Three Gaussian blurs at sigma = 15, 80, and 120 are applied. For each scale, the retinex output is the log-domain difference. The final texture map is the average across all three scales.', imageStepIndex: 4,
      latex: [
        '\\text{MSR}(x,y) = \\frac{1}{3} \\sum_{k} \\big[\\log(I(x,y) + 1) - \\log(G_{\\sigma_k} * I(x,y) + 1)\\big]',
      ] },
    { file: 'illumination.py', title: 'Step 4: Spherical Harmonics Lighting', desc: 'A 9-term spherical harmonic (SH) basis is computed from the surface normals. Each basis function captures a different lighting direction. A least-squares system is solved per color channel. The result splits into ambient light (from the zeroth harmonic) and direct light (remaining harmonics).', imageStepIndex: 5,
      latex: [
        '\\boldsymbol{\\gamma} = (A^\\top A + \\lambda I)^{-1} \\cdot A^\\top \\cdot \\frac{\\text{Image}}{\\text{Texture}}',
        '\\text{Ambient} = Y_0 \\cdot \\gamma_0',
        '\\text{Direct} = \\text{Illumination} - \\text{Ambient}',
      ] },
    { file: 'specular.py', title: 'Step 5: Specular Residual Extraction (SPR)', desc: 'The Lambertian rendering model assumes surfaces reflect light diffusely. Any leftover light after subtracting the Lambertian prediction is the specular residual. In real photos, specular highlights are physically consistent. In deepfakes, the GAN produces inconsistent specular patterns.', imageStepIndex: 7,
      latex: [
        '\\text{Lambertian} = (\\text{Ambient} + \\text{Direct}) \\cdot e^{\\text{Texture}}',
        '\\text{SPR} = \\frac{\\text{Image}}{255} - \\text{Lambertian}',
        '\\text{SPR} = \\max(\\text{SPR},\\; 0)',
      ] },
    { file: 'sri_net.py', title: 'Step 6: Specular Anomaly Score', desc: `The texture map and the specular residual are flattened into 1D vectors. The Normalized Cross-Correlation (NCC) between them is computed. A high NCC means the specular residual looks too similar to the texture, which is a sign of synthetic generation. ${df.specular !== undefined ? `Calculated Specular Anomaly Score = ${fmt(df.specular)}` : ''}`,
      outputLabel: 'Specular score', outputKey: 'specular',
      latex: [
        '\\text{NCC} = \\text{corr}(\\text{texture}_{\\text{flat}},\\; \\text{spr}_{\\text{flat}})',
        `\\text{anomaly} = \\sigma\\big(15 \\cdot (\\text{NCC} - 0.30)\\big) = ${fmt(df.specular)}`,
      ] },
    { file: 'frequency.py', title: 'Step 7: Frequency Domain Analysis (FFT)', desc: `The face crop is converted to grayscale and a 2D Fast Fourier Transform (FFT) is applied. The power spectrum is organized into a radial profile. The ratio of high-frequency energy to low-frequency energy (HFER) is calculated. GANs produce images with abnormal frequency distributions. ${df.frequency !== undefined ? `Calculated Frequency Anomaly Score = ${fmt(df.frequency)}` : ''}`,
      outputLabel: 'Frequency score', outputKey: 'frequency',
      latex: [
        '\\text{Power} = |\\mathcal{F}_{\\text{shifted}}|^2',
        '\\text{HFER} = \\frac{\\sum \\text{power}_{\\text{high}}}{\\sum \\text{power}_{\\text{low}}}',
        `\\text{anomaly} = \\sigma\\big(3.0 \\cdot (\\log_{10}(\\text{HFER}) + 4.5)\\big) = ${fmt(df.frequency)}`,
      ] },
    { file: 'patch_analysis.py', title: 'Step 8: Patch Noise Consistency (PRNU)', desc: `The Laplacian operator extracts high-frequency noise from the grayscale image. The image is divided into a 4x4 grid (16 patches). For each patch, the signal-to-noise ratio (SNR) is computed. The coefficient of variation (CV) across all patches measures consistency. Real cameras produce uniform noise; deepfakes show inconsistent patch noise. ${df.patch_consistency !== undefined ? `Calculated Patch Anomaly Score = ${fmt(df.patch_consistency)}` : ''}`,
      outputLabel: 'Patch score', outputKey: 'patch_consistency',
      latex: [
        '\\text{SNR}_i = \\frac{\\text{std}(\\text{patch}_i)}{\\text{mean}(\\text{patch}_i)}',
        '\\text{CV} = \\frac{\\text{std}(\\text{SNR}_{\\text{all}})}{\\text{mean}(\\text{SNR}_{\\text{all}})}',
        `\\text{anomaly} = \\sigma\\big(20 \\cdot (\\text{CV} - 0.25)\\big) = ${fmt(df.patch_consistency)}`,
      ] },
    { file: 'topology.py', title: 'Step 9: Topological Complexity Analysis', desc: `The specular residual magnitude is thresholded at three levels (64, 128, 192). At each level, the number of connected components and holes is counted. A weighted sum gives topological complexity. Deepfakes produce more complex, fragmented specular patterns. ${df.topology !== undefined ? `Calculated Topology Anomaly Score = ${fmt(df.topology)}` : ''}`,
      outputLabel: 'Topology score', outputKey: 'topology',
      latex: [
        'C = 0.20 C_{\\text{low}} + 0.35 C_{\\text{mid}} + 0.45 C_{\\text{high}} + 0.50 H_{\\text{low}} + 0.80 H_{\\text{mid}} + 1.10 H_{\\text{high}}',
        `\\text{anomaly} = \\sigma\\big(0.11 \\cdot (C - 18)\\big) = ${fmt(df.topology)}`,
      ] },
    { file: 'wavelet.py', title: 'Step 10: Wavelet Decomposition (DWT)', desc: `A 2-level Discrete Wavelet Transform using the Daubechies-2 wavelet decomposes the grayscale image. At each level, three detail sub-bands are produced: LH (horizontal), HL (vertical), and HH (diagonal). The HH band gets 1.5x weight because artifacts are most visible diagonally. ${df.wavelet_score !== undefined ? `Calculated Wavelet Anomaly Score = ${fmt(df.wavelet_score)}` : ''}`,
      outputLabel: 'Wavelet score', outputKey: 'wavelet_score',
      latex: [
        'E_{\\text{level}} = \\frac{E_{LH} + E_{HL} + 1.5 \\cdot E_{HH}}{3}',
        `\\text{anomaly} = \\sigma\\big(3 \\cdot (\\ln(1 + E_{\\text{avg}}) - 4.5)\\big) = ${fmt(df.wavelet_score)}`,
      ] },
    { file: 'vit_classifier.py', title: 'Step 11: Vision Transformer Classification (ViT)', desc: `The face crop is resized and passed through a pre-trained Vision Transformer model (Deep-Fake-Detector-v2). The model outputs logits for each class. Softmax converts logits to probabilities. ${df.vit_score !== undefined ? `Calculated ViT Anomaly Score = ${fmt(df.vit_score)}` : ''}`,
      outputLabel: 'ViT score', outputKey: 'vit_score',
      latex: [
        `P(\\text{fake}) = \\text{softmax}(\\text{logits})[\\text{fake\\_id}] = ${fmt(df.vit_score)}`,
      ] },
    { file: 'sri_net.py', title: 'Step 12: Noisy-OR Signal Fusion', desc: 'All six anomaly scores are combined using the Noisy-OR probabilistic fusion model. Each signal is treated as an independent detector. The probability that ALL signals indicate the image is real is the product of individual real probabilities. A sigmoid calibrates this ensemble score.',
      outputLabel: 'Ensemble', outputKey: 'physics_ensemble',
      latex: [
        'P(\\text{all real}) = \\prod_{i=1}^{6} \\big(1 - w_i \\cdot s_i\\big)',
        `\\text{ensemble} = 1 - P(\\text{all real}) = ${fmt(sb?.physics_ensemble)}`,
        `\\text{heuristic} = \\sigma\\big(8 \\cdot (\\text{ensemble} - 0.40)\\big) = ${fmt(df.heuristic_score)}`,
      ],
      formulaComputed: `Weights: spec=0.10, freq=0.18, topo=0.22, patch=0.22, wavelet=0.15, vit=0.13` },
    { file: 'evolution.py', title: 'Step 13: Calibration Model Blending', desc: 'A machine learning model is trained on the feature vector using seed data and user feedback. The learned score is blended with the heuristic using an adaptive weight that increases with feedback.',
      outputLabel: 'Final score', outputKey: 'final_score',
      latex: [
        'w = \\begin{cases} 0 & \\text{if feedback} < 15 \\\\ \\text{clamp}(0.10 + 0.02(n-15),\\; 0,\\; 0.80) & \\text{otherwise} \\end{cases}',
        '\\text{final} = w \\cdot \\text{learned} + (1 - w) \\cdot \\text{heuristic}',
        `= ${fmt(cb.model_weight)} \\times ${fmt(cb.learned_score)} + ${fmt(typeof cb.model_weight === 'number' ? 1 - cb.model_weight : undefined)} \\times ${fmt(cb.heuristic_score)} = ${fmt(df.final_score)}`,
      ] },
    { file: 'pipeline.py', title: 'Step 14: Final Verdict', desc: 'The final score is compared against a threshold of 0.50. If the score is greater than or equal to 0.50, the image is classified as Fake. Otherwise, it is classified as Real.',
      imageStepIndex: 8,
      latex: [
        '\\text{verdict} = \\begin{cases} \\text{Fake} & \\text{if score} \\geq 0.50 \\\\ \\text{Real} & \\text{if score} < 0.50 \\end{cases}',
        `\\text{confidence} = |\\text{score} - 0.50| \\times 2 = |${fmt(df.final_score)} - 0.50| \\times 2 = ${toPercent(r.confidence).replace('%', '\\\\%')}`,
      ],
      formulaComputed: `Final score = ${fmt(df.final_score)}, Verdict = ${r.verdict}` },
  ];
}

export function getFallbackSteps(r: AnalysisResponse): PipelineStepDef[] {
  const fb = r.fallback_breakdown || {};
  const df = r.decision_factors || {};
  const cb = r.calibration_breakdown || {};
  return [
    { file: 'face_detector.py', title: 'Step 1: Face Detection', desc: `MTCNN found no faces meeting the criteria (minimum 64x64 pixels, at least 90% confidence). Total faces detected: ${r.faces_detected || 0}. The system switches to full-image fallback mode, which uses frequency, wavelet, and ViT signals only.`, imageStepIndex: 1, outputLabel: 'Faces', outputKey: 'faces_detected' },
    { file: 'frequency.py', title: 'Step 2: Frequency Analysis', desc: `The image is resized to 256x256 pixels. A 2D FFT is applied to the grayscale version. The radial power spectrum is computed and the high-frequency energy ratio (HFER) is calculated. ${fb.frequency !== undefined ? `Calculated Frequency Score = ${fmt(fb.frequency)}` : ''}`,
      imageStepIndex: 2, outputLabel: 'Frequency', outputKey: 'frequency',
      latex: [
        '\\text{HFER} = \\frac{\\sum \\text{high\\_freq}}{\\sum \\text{low\\_freq}}',
        `\\text{anomaly} = \\sigma\\big(3.0 \\cdot (\\log_{10}(\\text{HFER}) + 4.5)\\big) = ${fmt(fb.frequency)}`,
      ] },
    { file: 'wavelet.py', title: 'Step 3: Wavelet Analysis', desc: `DWT (Daubechies-2, level 2) is applied to the full image. The energy in the high-frequency detail sub-bands is computed. ${fb.wavelet !== undefined ? `Calculated Wavelet Score = ${fmt(fb.wavelet)}` : ''}`,
      imageStepIndex: 3, outputLabel: 'Wavelet', outputKey: 'wavelet',
      latex: [
        `\\text{anomaly} = \\sigma\\big(3 \\cdot (\\ln(1 + E_{\\text{avg}}) - 4.5)\\big) = ${fmt(fb.wavelet)}`,
      ] },
    { file: 'vit_classifier.py', title: 'Step 4: ViT Classification', desc: `The full image (not a face crop) is passed through the Vision Transformer model. The softmax probability for the Fake class is returned. ${fb.vit_score !== undefined ? `Calculated ViT Score = ${fmt(fb.vit_score)}` : ''}`,
      outputLabel: 'ViT Score', outputKey: 'vit_score',
      latex: [
        `P(\\text{fake}) = \\text{softmax}(\\text{logits})[\\text{fake\\_id}] = ${fmt(fb.vit_score)}`,
      ] },
    { file: 'pipeline.py', title: 'Step 5: Weighted Combination', desc: 'The three signals are combined using fixed weights. The combined score is then passed through a sigmoid function to produce the heuristic score.',
      imageStepIndex: 4, outputLabel: 'Combined', outputKey: 'combined',
      latex: [
        `\\text{combined} = 0.30 \\cdot ${fmt(fb.frequency)} + 0.30 \\cdot ${fmt(fb.wavelet)} + 0.40 \\cdot ${fmt(fb.vit_score)} = ${fmt(fb.combined)}`,
        `\\text{heuristic} = \\sigma\\big(8 \\cdot (${fmt(fb.combined)} - 0.45)\\big) = ${fmt(fb.calibrated)}`,
      ] },
    { file: 'evolution.py', title: 'Step 6: Calibration Model', desc: 'A learned model blends with the heuristic. The weight is adaptive based on the amount of feedback data available.',
      outputLabel: 'Final score', outputKey: 'final_score',
      latex: [
        'w = \\begin{cases} 0 & \\text{if feedback} < 15 \\\\ \\text{clamp}(0.10 + 0.02(n-15),\\; 0,\\; 0.80) & \\text{otherwise} \\end{cases}',
        `\\text{final} = ${fmt(cb.model_weight)} \\times ${fmt(cb.learned_score)} + ${fmt(typeof cb.model_weight === 'number' ? 1 - cb.model_weight : undefined)} \\times ${fmt(cb.heuristic_score)} = ${fmt(df.final_score)}`,
      ] },
    { file: 'pipeline.py', title: 'Step 7: Final Verdict', desc: 'The final score is compared to the threshold of 0.50.',
      latex: [
        '\\text{verdict} = \\begin{cases} \\text{Fake} & \\text{if score} \\geq 0.50 \\\\ \\text{Real} & \\text{if score} < 0.50 \\end{cases}',
        `\\text{confidence} = |${fmt(df.final_score)} - 0.50| \\times 2 = ${toPercent(r.confidence).replace('%', '\\\\%')}`,
      ],
      formulaComputed: `score = ${fmt(df.final_score)}, Verdict = ${r.verdict}` },
  ];
}

export function getVideoPhysicsSteps(r: AnalysisResponse): PipelineStepDef[] {
  const fv = r.feature_vector || [0,0,0,0,0,0,0,0,0,0,0];
  const df = r.decision_factors || {};
  const cb = r.calibration_breakdown || {};
  return [
    { file: 'video_ingest.py', title: 'Step 1: Adaptive Frame Sampling', desc: 'The video is ingested and frames are sampled adaptively based on motion heuristics. FPS is normalized and quality metrics (blur, resolution) are computed to produce the video_quality index.',
      latex: [
        '\\text{quality} = 0.5 \\cdot \\text{resolution} + 0.3 \\cdot \\text{sharpness} + 0.2 \\cdot \\text{blockiness}',
        `\\text{video\\_quality} = ${fmt(fv[10])}`
      ] },
    { file: 'face_tracker.py', title: 'Step 2: Face Gateway (Multi-Frame Init)', desc: 'Scans the first 10 frames to find and lock onto the main face. If found, routes to Branch A (Face-Local Physics) and Branch B (Universal). If not found, routes to Branch B only (Fallback Mode).',
      latex: [
        `\\text{face\\_present} = ${fv[8] > 0.5 ? '1.0' : '0.0'}`,
        `\\text{avg\\_face\\_size} = ${fmt(fv[9])}`
      ] },
    { file: 'optical_boundary.py', title: 'Step 3: Optical Flow Boundary Analysis', desc: 'Dense optical flow detects microscopic motion slips at the jawline/hairline boundaries. The velocity gradient magnitude in the boundary ring is compared to the inner face.',
      latex: [
        '\\nabla \\mathbf{v} = \\sqrt{(\\partial v_x/\\partial x)^2 + (\\partial v_y/\\partial y)^2}',
        `\\text{anomaly} = \\sigma(4.0 \\cdot (\\text{ratio} - 1.8)) = ${fmt(fv[0])}`
      ] },
    { file: 'rppg_signal.py', title: 'Step 4: Biological Consistency (rPPG)', desc: 'Extracts Blood Volume Pulse from face and neck using the POS algorithm. Pearson correlation between the two BVP signals measures biological consistency. Uncorrelated pulses indicate a face swap.',
      latex: [
        '\\text{BVP} = X - \\alpha Y, \\quad X = 3R - 2G, \\quad Y = 1.5R + G - 1.5B',
        '\\text{correlation} = \\text{corr}(\\text{BVP}_{\\text{face}}, \\text{BVP}_{\\text{neck}})',
        `\\text{anomaly} = \\sigma(-8 \\cdot (\\text{corr} - 0.2)) = ${fmt(fv[1])}`
      ] },
    { file: 'temporal_lighting.py', title: 'Step 5: Temporal Illumination Jitter', desc: 'Spherical Harmonics coefficients are tracked across frames. Higher-order harmonics (directional light) are weighted more heavily since ambient light can flicker naturally.',
      latex: [
        '\\text{jitter} = \\frac{1}{T-1} \\sum_t \\|W \\cdot \\Delta \\boldsymbol{\\gamma}_t\\|',
        `\\text{anomaly} = \\sigma(8.0 \\cdot (\\text{jitter} - 0.35)) = ${fmt(fv[2])}`
      ] },
    { file: 'temporal_fft.py', title: 'Step 6: Temporal FFT Flicker Detection', desc: 'A 1D FFT is computed along the time axis for each pixel. The ratio of high-frequency to total energy reveals generative flicker invisible to the eye.',
      latex: [
        '\\text{HF\\_ratio} = \\frac{\\sum |F_{\\text{high}}|^2}{\\sum |F_{\\text{all}}|^2}',
        `\\text{anomaly} = \\sigma(15.0 \\cdot (\\text{HF\\_ratio} - 0.15)) = ${fmt(fv[3])}`
      ] },
    { file: 'wavelet_temporal.py', title: 'Step 7: Temporal Wavelet Grid Artifacts', desc: 'Haar wavelet HH-band diagonal artifacts from upsampling layers are tracked across frames. High temporal variance in the HH band indicates generative jitter.',
      latex: [
        '\\text{diff} = \\text{mean}(|\\text{HH}_t - \\text{HH}_{t-1}|)',
        `\\text{anomaly} = \\sigma(3.0 \\cdot (\\text{diff} - 1.5)) = ${fmt(fv[4])}`
      ] },
    { file: 'compression_residual.py', title: 'Step 8: Compression Residual Analysis', desc: 'Bilateral filtering extracts the noise residual per frame. Real compression noise is static across P-frames; generative noise flickers independently per frame.',
      latex: [
        '\\text{residual}_t = |I_t - \\text{bilateral}(I_t)|',
        `\\text{anomaly} = \\sigma(0.5 \\cdot (\\text{var}(\\text{residual}) - 10)) = ${fmt(fv[5])}`
      ] },
    { file: 'fullframe_temporal.py', title: 'Step 9: Global Scene Coherence', desc: 'Detects fully synthesized scenes (Sora, Veo, Kling) by measuring the spatial variance of the optical flow divergence. Real scenes have smooth gradients; AI scenes exhibit chaotic morphing.',
      latex: [
        '\\text{div}(\\mathbf{v}) = \\partial v_x/\\partial x + \\partial v_y/\\partial y',
        '\\text{chaos} = \\text{var}(\\text{div}(\\mathbf{v}))',
        `\\text{anomaly} = \\sigma(0.003 \\cdot (\\text{chaos} - 1200)) = ${fmt(fv[6])}`
      ] },
    { file: 'temporal_backbone.py', title: 'Step 10: Temporal Foundation Model (SSM)', desc: 'A deep State-Space Model wrapper identifies complex spatial-temporal anomalies with O(N) complexity. Processes the full frame sequence through a 3D convolutional proxy.',
      latex: [
        `\\text{score} = ${fmt(fv[7])}`
      ] },
    { file: 'video_sri_net.py', title: 'Step 11: Noisy-OR Fusion', desc: 'All 8 signals are fused probabilistically. Weights adapt dynamically based on face_present and avg_face_size. Small faces reduce physics weight; no face zeroes out Branch A.',
      latex: [
        'P(\\text{all\\_real}) = \\prod_{i=0}^{7} (1 - w_i \\cdot f_i)',
        `\\text{ensemble} = 1 - P(\\text{all\\_real}) = ${fmt(df.physics_ensemble)}`,
        `\\text{heuristic} = \\sigma(8 \\cdot (\\text{ensemble} - 0.40)) = ${fmt(df.heuristic_score)}`
      ] },
    { file: 'video_evolution.py', title: 'Step 12: Trust Calibration & Final Verdict', desc: 'Bayesian Beta-Binomial trust calibration blends the heuristic with historical feedback. Weight ramps from 0 to 0.80 after 15 feedback samples.',
      outputLabel: 'Final score', outputKey: 'final_score',
      latex: [
        'w = \\begin{cases} 0 & \\text{if feedback} < 15 \\\\ \\text{clamp}(0.10 + 0.02(n-15),\\; 0,\\; 0.80) & \\text{otherwise} \\end{cases}',
        '\\text{final} = w \\cdot \\text{trust} + (1 - w) \\cdot \\text{heuristic}',
        `= ${fmt(cb.model_weight)} \\times ${fmt(cb.learned_score)} + ${fmt(typeof cb.model_weight === 'number' ? 1 - cb.model_weight : undefined)} \\times ${fmt(cb.heuristic_score)} = ${fmt(df.final_score)}`,
        `\\text{verdict} = \\text{${r.verdict}}, \\quad \\text{confidence} = ${toPercent(r.confidence).replace('%', '\\\\%')}`
      ] }
  ];
}
