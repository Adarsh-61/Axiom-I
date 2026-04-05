'use client';

import Image from 'next/image';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

interface SignalBreakdown {
  specular: number;
  frequency: number;
  topology: number;
  patch_consistency: number;
  wavelet_score: number;
  vit_score: number;
  physics_ensemble: number;
  raw_fusion: number;
  calibrated: number;
}

interface FaceResult {
  bbox: number[];
  confidence: number;
  verdict: string;
  score: number;
  signal_breakdown?: SignalBreakdown;
}

interface VisualizationStep {
  step: number;
  label: string;
  data: string;
}

interface AnalysisResponse {
  verdict: string;
  confidence: number;
  faces_detected: number;
  faces: FaceResult[];
  steps: VisualizationStep[];
  full_image_score?: number;
  analysis_mode?: string;
  fallback_breakdown?: Record<string, number>;
  feature_vector?: number[];
  calibration_breakdown?: Record<string, number>;
  quality_metrics?: Record<string, number>;
  process_inputs?: {
    image_shape?: number[];
    analysis_mode?: string;
    components?: string[];
    feature_names?: string[];
  };
  decision_factors?: Record<string, number>;
  explanation?: string[];
  error?: string;
}

interface CalibrationBin {
  bin_start: number;
  bin_end: number;
  count: number;
  accuracy: number;
  confidence: number;
  gap: number;
}

interface FeedbackDiagnosticsResponse {
  confusion_matrix: {
    TP: number;
    TN: number;
    FP: number;
    FN: number;
    total: number;
  };
  calibration_metrics: {
    total_samples: number;
    brier_score: number;
    log_loss: number;
    ece: number;
    mce: number;
    mean_confidence: number;
    mean_accuracy: number;
    overconfidence_gap: number;
    bins?: CalibrationBin[];
    [key: string]: unknown;
  };
  calibration_history: Array<Record<string, unknown>>;
  feedback_summary: {
    total_feedback_records: number;
    training_eligible_records: number;
    training_excluded_records: number;
    training_exclusion_reasons: Record<string, number>;
    trust_summary?: {
      tracked_users: number;
      mean_trust_score: number | null;
      min_trust_score: number | null;
      max_trust_score: number | null;
      mean_sample_weight: number | null;
      [key: string]: unknown;
    };
  };
}

interface FeedbackSubmitResponse {
  status: string;
  message: string;
  confusion_matrix: {
    TP: number;
    TN: number;
    FP: number;
    FN: number;
    total?: number;
  };
  training_eligible: boolean;
  training_exclusion_reason?: string | null;
  calibration_metrics?: Record<string, unknown>;
  user_trust_score?: number;
  user_sample_weight?: number;
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';
const FEEDBACK_CLIENT_ID_KEY = 'axiom_feedback_client_id';

const getOrCreateClientId = (): string => {
  if (typeof window === 'undefined') {
    return 'anonymous';
  }

  const existing = window.localStorage.getItem(FEEDBACK_CLIENT_ID_KEY);
  if (existing && existing.trim().length > 0) {
    return existing;
  }

  const created =
    typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `client_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;

  window.localStorage.setItem(FEEDBACK_CLIENT_ID_KEY, created);
  return created;
};

const toPercent = (value: number | undefined): string => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '-';
  }
  return `${(value * 100).toFixed(1)}%`;
};

const toFixed = (value: number | undefined, digits = 4): string => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '-';
  }
  return value.toFixed(digits);
};

const toDisplay = (value: unknown): string => {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      return '-';
    }
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }

  if (typeof value === 'string') {
    return value;
  }

  if (Array.isArray(value)) {
    return value.length > 0 ? value.join(', ') : '-';
  }

  if (value === null || value === undefined) {
    return '-';
  }

  return JSON.stringify(value);
};

const labelOverrides: Record<string, string> = {
  ece: 'ECE',
  fn: 'FN',
  fp: 'FP',
  mce: 'MCE',
  tn: 'TN',
  tp: 'TP',
  vit: 'ViT',
};

const formatLabel = (raw: string): string => {
  const normalized = raw.trim();
  if (!normalized) {
    return '-';
  }

  const lower = normalized.toLowerCase();
  if (labelOverrides[lower]) {
    return labelOverrides[lower];
  }

  return normalized
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
    .replace(/\bVit\b/g, 'ViT');
};

const toFileSize = (bytes: number): string => {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '-';
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
};

const verdictClassName = (value: string | undefined): string => {
  const normalized = (value || '').trim().toLowerCase();
  if (normalized === 'real') {
    return 'textReal';
  }
  if (normalized === 'fake') {
    return 'textFake';
  }
  return '';
};

const percentLikeKeys = new Set([
  'confidence',
  'accuracy',
  'mean_confidence',
  'mean_accuracy',
  'overconfidence_gap',
  'mean_trust_score',
  'min_trust_score',
  'max_trust_score',
  'trust_score',
]);

const formatMetricValue = (key: string, value: unknown): string => {
  if (typeof value === 'number' && percentLikeKeys.has(key.toLowerCase())) {
    return toPercent(value);
  }
  return toDisplay(value);
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);
  const [diagnostics, setDiagnostics] = useState<FeedbackDiagnosticsResponse | null>(null);
  const [feedbackStatus, setFeedbackStatus] = useState<{
    message: string;
    trainingEligible: boolean;
    exclusionReason?: string | null;
    trustScore?: number;
    sampleWeight?: number;
  } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDiagnostics = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/feedback/metrics`);
      if (!response.ok) {
        return;
      }
      const payload = (await response.json()) as FeedbackDiagnosticsResponse;
      setDiagnostics(payload);
    } catch {
      return;
    }
  }, []);

  useEffect(() => {
    void fetchDiagnostics();
  }, [fetchDiagnostics]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const processComponents = useMemo(() => {
    return result?.process_inputs?.components ?? [];
  }, [result]);

  const processFeatureNames = useMemo(() => {
    return result?.process_inputs?.feature_names ?? [];
  }, [result]);

  const qualityMetricEntries = useMemo(() => {
    return Object.entries(result?.quality_metrics || {});
  }, [result]);

  const decisionFactorEntries = useMemo(() => {
    return Object.entries(result?.decision_factors || {});
  }, [result]);

  const calibrationBreakdownEntries = useMemo(() => {
    return Object.entries(result?.calibration_breakdown || {});
  }, [result]);

  const fallbackBreakdownEntries = useMemo(() => {
    return Object.entries(result?.fallback_breakdown || {});
  }, [result]);

  const featureVectorRows = useMemo(() => {
    const featureVector = result?.feature_vector || [];
    return featureVector.map((value, index) => ({
      index,
      name: processFeatureNames[index] || `feature_${index + 1}`,
      value,
    }));
  }, [result, processFeatureNames]);

  const diagnosticsCalibrationEntries = useMemo(() => {
    if (!diagnostics) {
      return [] as Array<[string, unknown]>;
    }

    return Object.entries(diagnostics.calibration_metrics).filter(([key]) => key !== 'bins');
  }, [diagnostics]);

  const diagnosticsBins = useMemo(() => {
    if (!diagnostics || !Array.isArray(diagnostics.calibration_metrics.bins)) {
      return [] as CalibrationBin[];
    }

    return diagnostics.calibration_metrics.bins;
  }, [diagnostics]);

  const exclusionReasonEntries = useMemo(() => {
    return Object.entries(diagnostics?.feedback_summary.training_exclusion_reasons || {});
  }, [diagnostics]);

  const trustSummaryEntries = useMemo(() => {
    if (!diagnostics?.feedback_summary.trust_summary) {
      return [] as Array<[string, unknown]>;
    }
    return Object.entries(diagnostics.feedback_summary.trust_summary);
  }, [diagnostics]);

  const calibrationHistoryRows = useMemo(() => {
    if (!diagnostics?.calibration_history) {
      return [] as Array<Record<string, unknown>>;
    }
    return diagnostics.calibration_history.slice(-10).reverse();
  }, [diagnostics]);

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(false);
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      handleFileSelection(event.dataTransfer.files[0]);
    }
  };

  const handleFileSelection = (selectedFile: File) => {
    setError(null);
    setResult(null);
    setFeedbackStatus(null);

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError('File size must be 10MB or less.');
      return;
    }

    const mimeType = (selectedFile.type || '').toLowerCase();
    const isImageMime =
      mimeType.startsWith('image/') ||
      mimeType === 'application/octet-stream' ||
      mimeType === '';

    if (!isImageMime) {
      setError('Only image files are supported.');
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
  };

  const runAnalysis = async () => {
    if (!file) {
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setFeedbackStatus(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE}/api/v1/analyze`, {
        method: 'POST',
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.detail || payload?.error || 'Analysis failed.');
      }

      setResult(payload as AnalysisResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetAll = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setFeedbackStatus(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const submitFeedback = async (userTruth: 'Real' | 'Fake') => {
    if (!result) {
      return;
    }

    const calibrated = result.calibration_breakdown?.calibrated_score;
    const faceScore = result.faces?.[0]?.score;
    const fallbackScore = result.fallback_breakdown?.calibrated;
    const effectiveScore =
      typeof calibrated === 'number'
        ? calibrated
        : typeof faceScore === 'number'
          ? faceScore
          : typeof fallbackScore === 'number'
            ? fallbackScore
            : typeof result.full_image_score === 'number'
              ? result.full_image_score
              : 0.5;

    const featureVector = Array.isArray(result.feature_vector) && result.feature_vector.length > 0
      ? result.feature_vector
      : null;

    setIsSubmittingFeedback(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/v1/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          full_image_score: effectiveScore,
          original_prediction: result.verdict,
          user_truth: userTruth,
          feature_vector: featureVector,
          user_id: getOrCreateClientId(),
        }),
      });

      const payload = (await response.json()) as FeedbackSubmitResponse;
      if (!response.ok) {
        throw new Error((payload as unknown as { detail?: string }).detail || 'Feedback submission failed.');
      }

      setFeedbackStatus({
        message: payload.message,
        trainingEligible: payload.training_eligible,
        exclusionReason: payload.training_exclusion_reason,
        trustScore: payload.user_trust_score,
        sampleWeight: payload.user_sample_weight,
      });

      await fetchDiagnostics();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Feedback submission failed.');
    } finally {
      setIsSubmittingFeedback(false);
    }
  };

  const matrix = diagnostics?.confusion_matrix;
  const tn = matrix?.TN ?? 0;
  const fp = matrix?.FP ?? 0;
  const fn = matrix?.FN ?? 0;
  const tp = matrix?.TP ?? 0;
  const rowRealTotal = tn + fp;
  const rowFakeTotal = fn + tp;
  const colRealTotal = tn + fn;
  const colFakeTotal = fp + tp;

  return (
    <div className="page">
      <header className="header">
        <h1 className="title">Axiom-I</h1>
      </header>

      {error && <div className="errorBox">{error}</div>}

      <section className="card workspaceCard">
        <div className="cardHead">
          <div>
            <h2 className="cardTitle">Analysis Workspace</h2>
            <p className="workspaceSubtitle">Image Forensic</p>
          </div>
        </div>

        <div className="workspaceGrid">
          <div className="workspacePane">
            <label
              className={`uploadArea ${isDragging ? 'active' : ''}`}
              htmlFor="upload-input"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              {!previewUrl ? (
                <>
                  <div className="uploadIconWrap">
                    <svg className="uploadIcon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.6} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                  </div>
                  <span className="uploadTitle">Upload Image</span>
                  <span className="uploadSubtitle">Any image format up to 10MB</span>
                </>
              ) : (
                <div className="previewBlock">
                  <div className="previewInfo">
                    <strong>{file?.name || 'Selected image'}</strong>
                    <span>{toFileSize(file?.size || 0)}</span>
                  </div>
                  <div className="previewWrap">
                    <Image src={previewUrl} alt="Selected image" width={1200} height={800} className="previewImage" unoptimized />
                  </div>
                </div>
              )}
            </label>

            <input
              id="upload-input"
              ref={fileInputRef}
              className="hidden"
              type="file"
              accept="image/*"
              onChange={(event) => {
                if (event.target.files && event.target.files.length > 0) {
                  handleFileSelection(event.target.files[0]);
                }
              }}
            />

            <div className="rowButtons">
              <button className="btn" onClick={resetAll} disabled={isAnalyzing || isSubmittingFeedback}>Reset</button>
              <button className="btn primary" onClick={runAnalysis} disabled={isAnalyzing || !file}>
                {isAnalyzing ? 'Analyzing...' : 'Analyze'}
              </button>
            </div>
          </div>

          <div className="workspacePane summaryPane">
            <h3 className="subTitle">Result Summary</h3>
            <div className="tableWrap">
              <table className="table summaryTable">
                <tbody>
                  <tr>
                    <th>File</th>
                    <td>{file?.name || '-'}</td>
                  </tr>
                  <tr>
                    <th>Verdict</th>
                    <td className={verdictClassName(result?.verdict)}>{result?.verdict || '-'}</td>
                  </tr>
                  <tr>
                    <th>Confidence</th>
                    <td>{toPercent(result?.confidence)}</td>
                  </tr>
                  <tr>
                    <th>Analysis Mode</th>
                    <td>{result?.analysis_mode || '-'}</td>
                  </tr>
                  <tr>
                    <th>Faces Detected</th>
                    <td>{typeof result?.faces_detected === 'number' ? result.faces_detected : '-'}</td>
                  </tr>
                  <tr>
                    <th>Final Score</th>
                    <td>{toFixed(result?.full_image_score)}</td>
                  </tr>
                  <tr>
                    <th>Heuristic Score</th>
                    <td>{toFixed(result?.calibration_breakdown?.heuristic_score)}</td>
                  </tr>
                  <tr>
                    <th>Learned Score</th>
                    <td>{toFixed(result?.calibration_breakdown?.learned_score)}</td>
                  </tr>
                  <tr>
                    <th>Model Weight</th>
                    <td>{toFixed(result?.calibration_breakdown?.model_weight)}</td>
                  </tr>
                  <tr>
                    <th>Training Samples</th>
                    <td>{toDisplay(result?.calibration_breakdown?.training_samples)}</td>
                  </tr>
                  <tr>
                    <th>Feedback Samples</th>
                    <td>{toDisplay(result?.calibration_breakdown?.feedback_samples)}</td>
                  </tr>
                  <tr>
                    <th>Feedback Weight Mean</th>
                    <td>{toFixed(result?.calibration_breakdown?.feedback_weight_mean)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {result && (
        <>
          <section className="card">
            <h2 className="cardTitle">Face Outputs</h2>
            {result.faces.length > 0 ? (
              <div className="faceGrid">
                {result.faces.map((face, index) => (
                  <div key={`face-${index}`} className="faceCard">
                    <div className="faceHeader">
                      <strong>Face {index + 1}</strong>
                      <span className={verdictClassName(face.verdict)}>{face.verdict}</span>
                    </div>
                    <div className="tableWrap">
                      <table className="table">
                        <tbody>
                          <tr>
                            <th>Bounding Box</th>
                            <td>{Array.isArray(face.bbox) ? face.bbox.join(', ') : '-'}</td>
                          </tr>
                          <tr>
                            <th>Detection Confidence</th>
                            <td>{toPercent(face.confidence)}</td>
                          </tr>
                          <tr>
                            <th>Face Score</th>
                            <td>{toFixed(face.score)}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                    <h3 className="subTitle">Signal Breakdown</h3>
                    {face.signal_breakdown ? (
                      <div className="tableWrap">
                        <table className="table">
                          <tbody>
                            {Object.entries(face.signal_breakdown).map(([key, value]) => (
                              <tr key={`${index}-${key}`}>
                                <th>{formatLabel(key)}</th>
                                <td>{toFixed(value)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="emptyState">No signal breakdown available.</div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="emptyState">No face outputs available in this run.</div>
            )}
          </section>

          <section className="card">
            <h2 className="cardTitle">Inputs and Components</h2>
            <div className="metaGrid">
              <div className="metaPanel">
                <h3>Input Shape</h3>
                <p>{Array.isArray(result.process_inputs?.image_shape) ? result.process_inputs.image_shape.join(' x ') : '-'}</p>
              </div>
              <div className="metaPanel">
                <h3>Pipeline Mode</h3>
                <p>{result.process_inputs?.analysis_mode || result.analysis_mode || '-'}</p>
              </div>
            </div>
            <h3 className="subTitle">Components Used</h3>
            <div className="chips">
              {processComponents.length > 0 ? processComponents.map((component) => (
                <span key={component} className="chip">{formatLabel(component)}</span>
              )) : <span className="muted">No component data</span>}
            </div>
            <h3 className="subTitle">Feature Names</h3>
            <div className="chips">
              {processFeatureNames.length > 0 ? processFeatureNames.map((name) => (
                <span key={name} className="chip">{formatLabel(name)}</span>
              )) : <span className="muted">No feature names available</span>}
            </div>
          </section>

          <section className="card">
            <h2 className="cardTitle">Quality Metrics</h2>
            {qualityMetricEntries.length > 0 ? (
              <div className="tableWrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {qualityMetricEntries.map(([key, value]) => (
                      <tr key={key}>
                        <td>{formatLabel(key)}</td>
                        <td>{toFixed(value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="emptyState">No quality metric data available.</div>
            )}
          </section>

          <section className="card">
            <h2 className="cardTitle">Decision Factors</h2>
            {decisionFactorEntries.length > 0 ? (
              <div className="tableWrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Factor</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {decisionFactorEntries.map(([key, value]) => (
                      <tr key={key}>
                        <td>{formatLabel(key)}</td>
                        <td>{toFixed(value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="emptyState">No decision factors available.</div>
            )}
          </section>

          <section className="card">
            <h2 className="cardTitle">Calibration Breakdown</h2>
            {calibrationBreakdownEntries.length > 0 ? (
              <div className="tableWrap">
                <table className="table">
                  <tbody>
                    {calibrationBreakdownEntries.map(([key, value]) => (
                      <tr key={key}>
                        <th>{formatLabel(key)}</th>
                        <td>{toDisplay(value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="emptyState">No calibration breakdown available.</div>
            )}
          </section>

          <section className="card">
            <h2 className="cardTitle">Fallback Breakdown</h2>
            {fallbackBreakdownEntries.length > 0 ? (
              <div className="tableWrap">
                <table className="table">
                  <tbody>
                    {fallbackBreakdownEntries.map(([key, value]) => (
                      <tr key={key}>
                        <th>{formatLabel(key)}</th>
                        <td>{toFixed(value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="emptyState">Fallback breakdown is not present for this run.</div>
            )}
          </section>

          <section className="card">
            <h2 className="cardTitle">Feature Vector</h2>
            {featureVectorRows.length > 0 ? (
              <div className="tableWrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Index</th>
                      <th>Name</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {featureVectorRows.map((row) => (
                      <tr key={`fv-${row.index}`}>
                        <td>{row.index}</td>
                        <td>{formatLabel(row.name)}</td>
                        <td>{toFixed(row.value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="emptyState">No feature vector available.</div>
            )}
          </section>

          <section className="card">
            <h2 className="cardTitle">Feedback and Diagnostics</h2>
            <div className="feedbackActions">
              <button className="btn textReal" onClick={() => submitFeedback('Real')} disabled={isSubmittingFeedback}>Mark Real</button>
              <button className="btn textFake" onClick={() => submitFeedback('Fake')} disabled={isSubmittingFeedback}>Mark Fake</button>
            </div>

            {feedbackStatus && (
              <div className="statusBox">
                <div>{feedbackStatus.message}</div>
                <div>
                  Training Eligible: {feedbackStatus.trainingEligible ? 'Yes' : 'No'}
                  {!feedbackStatus.trainingEligible && feedbackStatus.exclusionReason ? ` (${feedbackStatus.exclusionReason})` : ''}
                </div>
                <div>
                  Trust Score: {typeof feedbackStatus.trustScore === 'number' ? toPercent(feedbackStatus.trustScore) : '-'}
                </div>
                <div>
                  Sample Weight: {typeof feedbackStatus.sampleWeight === 'number' ? toFixed(feedbackStatus.sampleWeight, 2) : '-'}
                </div>
              </div>
            )}

            {diagnostics ? (
              <div className="diagnosticsStack">
                <div className="tableWrap">
                  <table className="table matrixTable">
                    <thead>
                      <tr>
                        <th>Actual / Predicted</th>
                        <th>Real</th>
                        <th>Fake</th>
                        <th>Total</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <th className="textReal">Real</th>
                        <td>{tn}</td>
                        <td>{fp}</td>
                        <td>{rowRealTotal}</td>
                      </tr>
                      <tr>
                        <th className="textFake">Fake</th>
                        <td>{fn}</td>
                        <td>{tp}</td>
                        <td>{rowFakeTotal}</td>
                      </tr>
                      <tr>
                        <th>Total</th>
                        <td>{colRealTotal}</td>
                        <td>{colFakeTotal}</td>
                        <td>{matrix?.total ?? rowRealTotal + rowFakeTotal}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div className="tableWrap">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Calibration Metric</th>
                        <th>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {diagnosticsCalibrationEntries.map(([key, value]) => (
                        <tr key={`metric-${key}`}>
                          <td>{formatLabel(key)}</td>
                          <td>{formatMetricValue(key, value)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {diagnosticsBins.length > 0 && (
                  <div className="tableWrap">
                    <table className="table">
                      <thead>
                        <tr>
                          <th>Bin</th>
                          <th>Count</th>
                          <th>Accuracy</th>
                          <th>Confidence</th>
                          <th>Gap</th>
                        </tr>
                      </thead>
                      <tbody>
                        {diagnosticsBins.map((bin, index) => (
                          <tr key={`bin-${index}`}>
                            <td>{`${bin.bin_start.toFixed(2)} to ${bin.bin_end.toFixed(2)}`}</td>
                            <td>{bin.count}</td>
                            <td>{toPercent(bin.accuracy)}</td>
                            <td>{toPercent(bin.confidence)}</td>
                            <td>{toPercent(bin.gap)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                <div className="tableWrap">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Feedback Summary</th>
                        <th>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Total Feedback Records</td>
                        <td>{diagnostics.feedback_summary.total_feedback_records}</td>
                      </tr>
                      <tr>
                        <td>Training Eligible Records</td>
                        <td>{diagnostics.feedback_summary.training_eligible_records}</td>
                      </tr>
                      <tr>
                        <td>Training Excluded Records</td>
                        <td>{diagnostics.feedback_summary.training_excluded_records}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div className="tableWrap">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Exclusion Reason</th>
                        <th>Count</th>
                      </tr>
                    </thead>
                    <tbody>
                      {exclusionReasonEntries.length > 0 ? exclusionReasonEntries.map(([key, value]) => (
                        <tr key={`reason-${key}`}>
                          <td>{formatLabel(key)}</td>
                          <td>{value}</td>
                        </tr>
                      )) : (
                        <tr>
                          <td colSpan={2}>No exclusions recorded.</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                <div className="tableWrap">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Trust Summary</th>
                        <th>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {trustSummaryEntries.length > 0 ? trustSummaryEntries.map(([key, value]) => (
                        <tr key={`trust-${key}`}>
                          <td>{formatLabel(key)}</td>
                          <td>{formatMetricValue(key, value)}</td>
                        </tr>
                      )) : (
                        <tr>
                          <td colSpan={2}>No trust summary available.</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                <div className="tableWrap">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Timestamp</th>
                        <th>Samples</th>
                        <th>Brier</th>
                        <th>ECE</th>
                        <th>MCE</th>
                        <th>FP</th>
                        <th>FN</th>
                      </tr>
                    </thead>
                    <tbody>
                      {calibrationHistoryRows.length > 0 ? calibrationHistoryRows.map((row, index) => (
                        <tr key={`history-${index}`}>
                          <td>{toDisplay(row.timestamp)}</td>
                          <td>{toDisplay(row.total_samples)}</td>
                          <td>{toDisplay(row.brier_score)}</td>
                          <td>{toDisplay(row.ece)}</td>
                          <td>{toDisplay(row.mce)}</td>
                          <td>{toDisplay(row.fp)}</td>
                          <td>{toDisplay(row.fn)}</td>
                        </tr>
                      )) : (
                        <tr>
                          <td colSpan={7}>No calibration history available.</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="emptyState">No feedback diagnostics available.</div>
            )}
          </section>

          <section className="card">
            <h2 className="cardTitle">Process Steps</h2>
            {result.steps.length > 0 ? (
              <div className="stepsGrid">
                {result.steps.map((step) => (
                  <div key={step.step} className="stepCard">
                    <Image src={step.data} alt={step.label} width={320} height={180} className="stepImage" unoptimized />
                    <div className="stepMeta">
                      <strong>Step {step.step}</strong>
                      <span>{step.label}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="emptyState">No pipeline step visualizations available.</div>
            )}
          </section>

          <section className="card">
            <h2 className="cardTitle">How System Works</h2>
            <ul className="explainList">
              {(result.explanation || []).map((line, index) => (
                <li key={`explain-${index}`}>{line}</li>
              ))}
              <li>Decision rule: final_score &gt;= 0.50 means Fake, otherwise Real.</li>
              <li>Confidence formula: confidence = |final_score - 0.5| * 2.</li>
              <li>Calibration formula: final_score = model_weight * learned_score + (1 - model_weight) * heuristic_score.</li>
              <li>Pipeline mode in this run: {result.analysis_mode || '-'}</li>
            </ul>
          </section>
        </>
      )}
    </div>
  );
}
