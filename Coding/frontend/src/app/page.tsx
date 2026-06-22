'use client';

import Image from 'next/image';

import { useCallback, useEffect, useRef, useState, useMemo } from 'react';
import { Tex } from './Tex';
import {
  API_BASE, ANALYZE_TIMEOUT_MS, VIDEO_ANALYZE_TIMEOUT_MS, FEEDBACK_TIMEOUT_MS, METRICS_TIMEOUT_MS,
  AnalysisResponse, FeedbackDiagnosticsResponse, FeedbackSubmitResponse,
  PipelineStepDef, ErrorPayload,
  withTimeout, parseJsonSafe, extractErrorMessage, getOrCreateClientId,
  toPercent, toFixed, toFileSize, getFullPhysicsSteps, getFallbackSteps, getVideoPhysicsSteps,
} from './helpers';

const signalColor = (v: number) => v >= 0.6 ? 'danger' : v >= 0.3 ? 'mid' : 'safe';
const f = (v: number | undefined) => typeof v === 'number' ? v.toFixed(4) : '?';

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
    message: string; trainingEligible: boolean; exclusionReason?: string | null;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDiagnostics = useCallback(async () => {
    try {
      const res = await withTimeout(`${API_BASE}/api/v1/feedback/metrics`, { method: 'GET' }, METRICS_TIMEOUT_MS);
      if (!res.ok) return;
      const p = await parseJsonSafe<FeedbackDiagnosticsResponse>(res);
      if (p) setDiagnostics(p);
    } catch { /* silent */ }
  }, []);

  useEffect(() => { void fetchDiagnostics(); }, [fetchDiagnostics]);
  useEffect(() => { return () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }; }, [previewUrl]);

  const handleFileSelection = (selectedFile: File) => {
    setError(null); setResult(null); setFeedbackStatus(null);
    if (selectedFile.size > 50 * 1024 * 1024) { setError('File size must be 50 MB or less.'); return; }
    const mime = (selectedFile.type || '').toLowerCase();
    const name = selectedFile.name.toLowerCase();
    const isImage = mime.startsWith('image/') || name.endsWith('.jpg') || name.endsWith('.jpeg') || name.endsWith('.png') || name.endsWith('.webp');
    const isVideo = mime.startsWith('video/') || name.endsWith('.mp4') || name.endsWith('.webm') || name.endsWith('.mov') || name.endsWith('.avi');
    if (!isImage && !isVideo && mime !== 'application/octet-stream') { setError('Only image and video files are supported.'); return; }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(selectedFile); setPreviewUrl(URL.createObjectURL(selectedFile));
  };

  const runAnalysis = async () => {
    if (!file) return;
    setIsAnalyzing(true); setError(null); setFeedbackStatus(null);
    try {
      const fd = new FormData(); fd.append('file', file);
      const name = file.name.toLowerCase();
      const isVideo = file.type.startsWith('video/') || name.endsWith('.mp4') || name.endsWith('.webm') || name.endsWith('.mov') || name.endsWith('.avi');
      const endpoint = isVideo ? `${API_BASE}/api/v1/analyze/video` : `${API_BASE}/api/v1/analyze`;
      // Video analysis uses a longer timeout because the multi-frame pipeline is
      // significantly heavier, and a cold Hugging Face Space start can take up to a minute.
      const timeoutMs = isVideo ? VIDEO_ANALYZE_TIMEOUT_MS : ANALYZE_TIMEOUT_MS;
      const res = await withTimeout(endpoint, { method: 'POST', body: fd }, timeoutMs);
      const p = await parseJsonSafe<AnalysisResponse & ErrorPayload>(res);
      if (!res.ok) throw new Error(extractErrorMessage(p, 'Analysis failed.'));
      if (!p) throw new Error('Empty response.');
      setResult(p as AnalysisResponse);
    } catch (err) {
      const isTimeout = err instanceof DOMException && err.name === 'AbortError';
      setError(isTimeout
        ? 'Analysis timed out. Video analysis can take up to 3 minutes on first load. Please try again.'
        : err instanceof Error ? err.message : 'Analysis failed.');
    } finally { setIsAnalyzing(false); }
  };

  const resetAll = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null); setPreviewUrl(null); setResult(null); setError(null); setFeedbackStatus(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const submitFeedback = async (userTruth: 'Real' | 'Fake') => {
    if (!result) return;
    const cb = result.calibration_breakdown;
    const effectiveScore = typeof cb?.calibrated_score === 'number' ? cb.calibrated_score
      : typeof result.faces?.[0]?.score === 'number' ? result.faces[0].score
      : typeof result.full_image_score === 'number' ? result.full_image_score : 0.5;
    setIsSubmittingFeedback(true); setError(null);
    try {
      const isVideo = result.analysis_mode?.startsWith('video');
      const endpoint = isVideo ? `${API_BASE}/api/v1/feedback/video` : `${API_BASE}/api/v1/feedback`;
      const res = await withTimeout(endpoint, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(isVideo ? { video_id: result.video_id || '', is_correct: result.verdict === userTruth, feature_vector: result.feature_vector || [], user_rating: userTruth === 'Fake' ? 1.0 : 0.0 } : { full_image_score: effectiveScore, original_prediction: result.verdict, user_truth: userTruth, feature_vector: result.feature_vector || null, user_id: getOrCreateClientId() }),
      }, FEEDBACK_TIMEOUT_MS);

      const p = await parseJsonSafe<FeedbackSubmitResponse & ErrorPayload>(res);
      if (!res.ok) throw new Error(extractErrorMessage(p, 'Feedback failed.'));
      if (!p) throw new Error('Empty response.');
      if (isVideo) {
        setFeedbackStatus({
          message: p.message ?? 'Video feedback recorded.',
          trainingEligible: typeof p.training_eligible === 'boolean' ? p.training_eligible : true,
          exclusionReason: p.training_exclusion_reason ?? null,
        });
      } else {
        setFeedbackStatus({
          message: p.message,
          trainingEligible: p.training_eligible,
          exclusionReason: p.training_exclusion_reason,
        });
      }
      await fetchDiagnostics();
    } catch (err) { setError(err instanceof Error ? err.message : 'Feedback failed.'); }
    finally { setIsSubmittingFeedback(false); }
  };

  const pipelineSteps: PipelineStepDef[] = useMemo(() => {
    if (!result) return [];
    if (result.analysis_mode?.startsWith('video')) return getVideoPhysicsSteps(result);
    return result.analysis_mode === 'fallback' ? getFallbackSteps(result) : getFullPhysicsSteps(result);
  }, [result]);

  const signalScores = useMemo(() => {
    if (!result) return [];
    const df = result.decision_factors || {};
    const sb = result.faces?.[0]?.signal_breakdown;
    
    if (result.analysis_mode?.startsWith('video')) {
      const fv = result.feature_vector || [0,0,0,0,0,0,0,0,0,0,0];
      return [
        { name: 'Opt. Flow', value: fv[0], weight: 0.18 },
        { name: 'rPPG', value: fv[1], weight: 0.18 },
        { name: 'Lighting', value: fv[2], weight: 0.15 },
        { name: 'FFT', value: fv[3], weight: 0.10 },
        { name: 'Wavelet', value: fv[4], weight: 0.10 },
        { name: 'Compression', value: fv[5], weight: 0.05 },
        { name: 'Scene (Branch B)', value: fv[6], weight: 0.12 },
        { name: 'Mamba (Branch B)', value: fv[7], weight: 0.12 },
      ];
    }
    
    if (result.analysis_mode === 'fallback') {
      const fb = result.fallback_breakdown || {};
      return [
        { name: 'Frequency', value: fb.frequency ?? df.frequency ?? 0, weight: 0.30 },
        { name: 'Wavelet', value: fb.wavelet ?? df.wavelet_score ?? 0, weight: 0.30 },
        { name: 'ViT Score', value: fb.vit_score ?? df.vit_score ?? 0, weight: 0.40 },
      ];
    }
    return [
      { name: 'Specular', value: sb?.specular ?? df.specular ?? 0, weight: 0.10 },
      { name: 'Frequency', value: sb?.frequency ?? df.frequency ?? 0, weight: 0.18 },
      { name: 'Topology', value: sb?.topology ?? df.topology ?? 0, weight: 0.22 },
      { name: 'Patch (PRNU)', value: sb?.patch_consistency ?? df.patch_consistency ?? 0, weight: 0.22 },
      { name: 'Wavelet', value: sb?.wavelet_score ?? df.wavelet_score ?? 0, weight: 0.15 },
      { name: 'ViT Score', value: sb?.vit_score ?? df.vit_score ?? 0, weight: 0.13 },
    ];
  }, [result]);

  const getStepImage = (idx: number | undefined): string | null => {
    if (idx === undefined || !result?.steps) return null;
    return result.steps.find(s => s.step === idx)?.data || null;
  };

  const getOutputValue = (key: string | undefined): string => {
    if (!key || !result) return '';
    const df = result.decision_factors || {};
    const fb = result.fallback_breakdown || {};
    const v = (df as Record<string, number>)[key] ?? (fb as Record<string, number>)[key]
      ?? (key === 'faces_detected' ? result.faces_detected : undefined);
    return typeof v === 'number' ? v.toFixed(4) : String(v ?? '');
  };

  const verdictColorClass = result?.verdict?.toLowerCase() === 'real' ? 'textReal' : result?.verdict?.toLowerCase() === 'fake' ? 'textFake' : '';
  const matrix = diagnostics?.confusion_matrix;
  const isFullPhysics = result?.analysis_mode !== 'fallback';
  const df = result?.decision_factors || {};
  const sb = result?.faces?.[0]?.signal_breakdown;
  const cb = result?.calibration_breakdown || {};
  const fb = result?.fallback_breakdown || {};

  return (
    <div className="page">
      <header className="header">
        <h1 className="title">Axiom-I</h1>
        <p className="subtitle">Media Forensics Analysis System</p>
      </header>

      {error && <div className="errorBox">{error}</div>}

      {/* Upload and Summary */}
      <section className="card">
        <div className="cardTitle">Analysis Workspace</div>
        <div className="workspaceGrid">
          <div>
            <label className={`uploadArea ${isDragging ? 'active' : ''}`} htmlFor="upload-input"
              onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={e => { e.preventDefault(); setIsDragging(false); }}
              onDrop={e => { e.preventDefault(); setIsDragging(false); if (e.dataTransfer.files?.length) handleFileSelection(e.dataTransfer.files[0]); }}>
              {!previewUrl ? (<>
                <div className="uploadIconWrap">
                  <svg className="uploadIcon" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.6} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
                </div>
                <span className="uploadTitle">Upload Media</span>
                <span className="uploadSubtitle">Image or video (up to 50 MB)</span>
              </>) : (
                <div className="previewBlock">
                  <div className="previewInfo"><strong>{file?.name || 'Selected media'}</strong><span>{toFileSize(file?.size || 0)}</span></div>
                  <div className="previewWrap">
                    {file && (file.type.startsWith('video/') || file.name.toLowerCase().endsWith('.mp4') || file.name.toLowerCase().endsWith('.webm') || file.name.toLowerCase().endsWith('.mov') || file.name.toLowerCase().endsWith('.avi')) ? (
                      <video key={previewUrl} src={previewUrl} controls className="previewImage" style={{ maxHeight: 300, width: 'auto', display: 'block', margin: '0 auto' }}>
                        Your browser does not support the video tag.
                      </video>
                    ) : (
                      <Image src={previewUrl} alt="Selected media" width={1200} height={800} className="previewImage" unoptimized />
                    )}
                  </div>
                </div>
              )}
            </label>
            <input id="upload-input" ref={fileInputRef} className="hidden" type="file" accept="image/*,video/*" onChange={e => { if (e.target.files?.length) handleFileSelection(e.target.files[0]); }} />
            <div className="rowButtons" style={{ marginTop: 10 }}>
              <button className="btn" onClick={resetAll} disabled={isAnalyzing || isSubmittingFeedback}>Reset</button>
              <button className="btn primary" onClick={runAnalysis} disabled={isAnalyzing || !file}>{isAnalyzing ? 'Analyzing...' : 'Analyze'}</button>
            </div>
          </div>
          <div className="summaryPane">
            <h3 className="subTitle">Result Summary</h3>
            {result ? (
              <div className="tableWrap"><table className="table"><tbody>
                <tr><th>Verdict</th><td className={verdictColorClass} style={{ fontWeight: 700 }}>{result.verdict}</td></tr>
                <tr><th>Confidence</th><td>{toPercent(result.confidence)}</td></tr>
                <tr><th>Analysis Mode</th><td>{result.analysis_mode === 'fallback' ? 'Fallback (no face detected)' : result.analysis_mode === 'video_full' ? 'Video Physics (face detected)' : result.analysis_mode === 'video_fallback' ? 'Video Fallback (no face)' : 'Full Physics (face detected)'}</td></tr>
                <tr><th>Faces Detected</th><td>{result.faces_detected}</td></tr>
                <tr><th>Final Score</th><td>{toFixed(df.final_score ?? result.full_image_score)}</td></tr>
                <tr><th>Heuristic Score</th><td>{toFixed(cb.heuristic_score)}</td></tr>
                <tr><th>Learned Score</th><td>{toFixed(cb.learned_score)}</td></tr>
                <tr><th>Model Weight</th><td>{toFixed(cb.model_weight)}</td></tr>
              </tbody></table></div>
            ) : (
              <div className="emptyState">Upload an image or video and click Analyze to see results.</div>
            )}
          </div>
        </div>
      </section>

      {result && (<>
        {/* Signal Scores with color coding */}
        <section className="card">
          <div className="cardTitle">Signal Anomaly Scores</div>
          <p className="cardDesc">Each signal measures a different forensic property. Values closer to 1.0 indicate stronger evidence of manipulation. Green means safe, amber means moderate, and red means high anomaly.</p>
          <div className="signalGrid">
            {signalScores.map(s => {
              const cls = signalColor(s.value);
              return (
                <div key={s.name} className="signalRow">
                  <span className="signalName">{s.name} <span className="signalWeight">(w={s.weight})</span></span>
                  <div className="signalTrack"><div className={`signalFill ${cls}`} style={{ width: `${Math.min(Math.max(s.value * 100, 0), 100)}%` }} /></div>
                  <span className={`signalValue ${cls}`}>{s.value.toFixed(4)}</span>
                </div>
              );
            })}
          </div>
        </section>

        {/* Execution Pipeline with KaTeX */}
        <section className="card">
          <div className="cardTitle">Execution Pipeline: File by File</div>
          <p className="cardDesc">Complete execution flow of the analysis. Each step corresponds to a Python file. Data flows top to bottom.</p>
          <div className="pipelineFlow">
            {pipelineSteps.map((step, i) => {
              const img = getStepImage(step.imageStepIndex);
              const outVal = getOutputValue(step.outputKey);
              return (
                <div key={i} className="pipelineStep">
                  <div className="stepIndicator">
                    <div className="stepBadge">{i + 1}</div>
                    <div className="stepLine" />
                  </div>
                  <div className="stepContent">
                    <div className="stepHeader">
                      <span className="stepFileName">{step.file}</span>
                      <span className="stepTitle">{step.title}</span>
                    </div>
                    <div className="stepDesc">{step.desc}</div>
                    {img && (<div className="stepImageWrap"><Image src={img} alt={step.title} width={220} height={150} className="stepImage" unoptimized /></div>)}
                    {step.latex && step.latex.length > 0 && (
                      <div className="formulaBlock">
                        <span className="formulaLabel">Formula</span>
                        {step.latex.map((tex, j) => (
                          <div key={j} style={{ marginBottom: 4 }}><Tex math={tex} block /></div>
                        ))}
                      </div>
                    )}
                    {step.formulaComputed && (
                      <div className="formulaBlock">
                        <span className="formulaLabel">Computed</span>
                        <span style={{ fontFamily: 'var(--mono)', fontSize: '0.78rem', whiteSpace: 'pre-wrap' }}>{step.formulaComputed}</span>
                      </div>
                    )}
                    {step.outputLabel && outVal && (
                      <div className="stepOutput"><span className="stepOutputLabel">{step.outputLabel}:</span><span className="stepOutputValue">{outVal}</span></div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="card">
          <div className="cardTitle" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>How the decision is made</span>
            <button className="linkBtn" onClick={() => {
              sessionStorage.setItem('axiom_result', JSON.stringify(result));
              window.open('/math', '_blank');
            }}>
              View full mathematics
            </button>
          </div>
          <p className="cardDesc">Summary of the mathematical derivation showing how the final verdict of <strong className={verdictColorClass}>{result.verdict}</strong> with {toPercent(result.confidence)} confidence was computed. Click the button above to see the complete step-by-step calculations with all variable values.</p>

          {!result.analysis_mode?.startsWith('video') && (
            <div className="mathSection">
              <div className="mathSectionTitle">1. Sigmoid Function</div>
              <p className="mathText">All anomaly scores use the sigmoid function to map raw values to [0, 1]:</p>
              <div className="formulaBlock"><Tex math="\sigma(x) = \frac{1}{1 + e^{-x}}" block /></div>
            </div>
          )}

          {!result.analysis_mode?.startsWith('video') && isFullPhysics && (
            <>
              <div className="mathSection">
                <div className="mathSectionTitle">2. Individual Signal Scores</div>
                <p className="mathText">Each forensic module computes an anomaly score using shifted sigmoid functions:</p>
                <div className="formulaBlock">
                  <Tex math={`\\text{Specular} = \\sigma\\big(15 \\cdot (\\text{NCC} - 0.30)\\big) = ${f(df.specular)}`} block />
                  <Tex math={`\\text{Frequency} = \\sigma\\big(3.0 \\cdot (\\log_{10}(\\text{HFER}) + 4.5)\\big) = ${f(df.frequency)}`} block />
                  <Tex math={`\\text{Topology} = \\sigma\\big(0.11 \\cdot (C - 18)\\big) = ${f(df.topology)}`} block />
                  <Tex math={`\\text{Patch} = \\sigma\\big(20 \\cdot (\\text{CV} - 0.25)\\big) = ${f(df.patch_consistency)}`} block />
                  <Tex math={`\\text{Wavelet} = \\sigma\\big(3 \\cdot (\\ln(1 + E_{\\text{avg}}) - 4.5)\\big) = ${f(df.wavelet_score)}`} block />
                  <Tex math={`\\text{ViT} = \\text{softmax}(\\text{logits})[\\text{fake}] = ${f(df.vit_score)}`} block />
                </div>
              </div>

              <div className="mathSection">
                <div className="mathSectionTitle">3. Noisy-OR Fusion</div>
                <p className="mathText">The six signals are fused using the Noisy-OR probabilistic model. Each signal has an assigned weight. The probability that all signals indicate the image is real is the product of individual real probabilities:</p>
                <div className="formulaBlock">
                  <Tex math="P(\text{all real}) = \prod_{i=1}^{6} \big(1 - w_i \cdot s_i\big)" block />
                  <Tex math={`\\text{ensemble} = 1 - P(\\text{all real}) = ${f(sb?.physics_ensemble)}`} block />
                  <Tex math={`\\text{heuristic} = \\sigma\\big(8 \\cdot (\\text{ensemble} - 0.40)\\big) = ${f(df.heuristic_score)}`} block />
                </div>
                <p className="mathText">Weights: specular = 0.10, frequency = 0.18, topology = 0.22, patch = 0.22, wavelet = 0.15, ViT = 0.13</p>
              </div>
            </>
          )}

          {!result.analysis_mode?.startsWith('video') && !isFullPhysics && (
            <div className="mathSection">
              <div className="mathSectionTitle">2. Fallback Signal Fusion</div>
              <p className="mathText">Without a detected face, three signals are combined using fixed weights:</p>
              <div className="formulaBlock">
                <Tex math={`\\text{combined} = 0.30 \\cdot ${f(fb.frequency)} + 0.30 \\cdot ${f(fb.wavelet)} + 0.40 \\cdot ${f(fb.vit_score)} = ${f(fb.combined)}`} block />
                <Tex math={`\\text{heuristic} = \\sigma\\big(8 \\cdot (${f(fb.combined)} - 0.45)\\big) = ${f(fb.calibrated)}`} block />
              </div>
            </div>
          )}

          {!result.analysis_mode?.startsWith('video') && (
            <>
              <div className="mathSection">
                <div className="mathSectionTitle">{isFullPhysics ? '4' : '3'}. Calibration Blend</div>
                <p className="mathText">A machine learning model produces a learned score. This is blended with the heuristic using an adaptive weight that increases with more feedback data:</p>
                <div className="formulaBlock">
                  <Tex math="w = \begin{cases} 0 & \text{if feedback} < 15 \\ \text{clamp}(0.10 + 0.02 \cdot (n - 15),\; 0,\; 0.80) & \text{otherwise} \end{cases}" block />
                  <Tex math="\text{final} = w \cdot \text{learned} + (1 - w) \cdot \text{heuristic}" block />
                  <Tex math={`= ${f(cb.model_weight)} \\times ${f(cb.learned_score)} + ${f(typeof cb.model_weight === 'number' ? 1 - cb.model_weight : undefined)} \\times ${f(cb.heuristic_score)} = ${f(df.final_score)}`} block />
                </div>
              </div>

              <div className="mathSection">
                <div className="mathSectionTitle">{isFullPhysics ? '5' : '4'}. Final Decision</div>
                <div className="formulaBlock">
                  <Tex math='\text{verdict} = \begin{cases} \text{Fake} & \text{if } \text{score} \geq 0.50 \\ \text{Real} & \text{if } \text{score} < 0.50 \end{cases}' block />
                  <Tex math={`\\text{confidence} = |\\text{score} - 0.50| \\times 2 = |${f(df.final_score)} - 0.50| \\times 2 = ${toPercent(result.confidence)}`} block />
                </div>
                <p className="mathText">Result: <strong className={verdictColorClass}>{result.verdict}</strong> with {toPercent(result.confidence)} confidence (score = {f(df.final_score)})</p>
              </div>
            </>
          )}
        </section>

        {/* Feedback */}
        <section className="card">
          <div className="cardTitle">Feedback</div>
          <p className="cardDesc">Was the prediction correct? Your feedback improves the calibration model.</p>
          <div className="feedbackActions">
            <button className="btn textReal" onClick={() => submitFeedback('Real')} disabled={isSubmittingFeedback}>Mark as Real</button>
            <button className="btn textFake" onClick={() => submitFeedback('Fake')} disabled={isSubmittingFeedback}>Mark as Fake</button>
          </div>
          {feedbackStatus && (<div className="statusBox"><div>{feedbackStatus.message}</div><div>Training Eligible: {feedbackStatus.trainingEligible ? 'Yes' : `No (${feedbackStatus.exclusionReason || 'unknown'})`}</div></div>)}
        </section>

        {/* System Diagnostics - shows skeleton while loading, data once available */}
        <section className="card">
          <div className="cardTitle">System Diagnostics</div>
          {diagnostics ? (
            <>
              <div className="tableWrap"><table className="table matrixTable">
                <thead><tr><th>Actual / Predicted</th><th>Real</th><th>Fake</th></tr></thead>
                <tbody>
                  <tr><th className="textReal">Real</th><td>{matrix?.TN ?? 0}</td><td>{matrix?.FP ?? 0}</td></tr>
                  <tr><th className="textFake">Fake</th><td>{matrix?.FN ?? 0}</td><td>{matrix?.TP ?? 0}</td></tr>
                </tbody>
              </table></div>
              <p className="cardDesc">Total Feedback: {diagnostics.feedback_summary.total_feedback_records} | Training Eligible: {diagnostics.feedback_summary.training_eligible_records}</p>
            </>
          ) : (
            <div style={{ display: 'grid', gap: 6 }}>
              <div className="skeleton skeletonLine" style={{ width: '100%', height: 60 }} />
              <div className="skeleton skeletonLine" />
              <div className="skeleton skeletonLine" />
            </div>
          )}
        </section>
      </>)}

    </div>
  );
}
