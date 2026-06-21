'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Tex } from '../Tex';
import { AnalysisResponse, getVideoPhysicsSteps } from '../helpers';

const f = (v: number | undefined) => typeof v === 'number' ? v.toFixed(4) : '?';

export default function MathPage() {
  const router = useRouter();
  const [r, setR] = useState<AnalysisResponse | null>(null);

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem('axiom_result');
      if (raw) setR(JSON.parse(raw));
    } catch {}
  }, []);

  const df = r?.decision_factors || {};
  const cb = r?.calibration_breakdown || {} as Record<string, number>;
  const fb = r?.fallback_breakdown || {};
  const face = r?.faces?.[0];
  const sb = face?.signal_breakdown;
  const isVideo = r?.analysis_mode?.startsWith('video');
  const isPhysics = r?.analysis_mode !== 'fallback' && !isVideo;

  const renderVideoMath = () => {
    if (!r || !isVideo) return null;
    const steps = getVideoPhysicsSteps(r);
    return steps.map((step, idx) => (
      <div className="deepStep" key={`video_step_${idx}`}>
        <div className="deepStepHeader">
          <div className="deepStepNum">{idx + 1}</div>
          <span className="deepStepFile">{step.file}</span>
          <h2 className="deepStepTitle">{step.title}</h2>
        </div>
        <p className="deepDesc">{step.desc}</p>
        <div className="derivBlock">
          {step.latex?.map((lx, i) => (
            <Tex math={lx} block key={i} />
          ))}
        </div>
      </div>
    ));
  };

  return (
    <div className="mathPage">
      <div className="mathPageHeader">
        <button className="backLink" onClick={() => {
          if (typeof window !== 'undefined' && (window.opener || window.history.length === 1)) {
            window.close();
          } else {
            router.push('/');
          }
        }}>&#8592; Back to analysis workspace</button>
        <h1 className="mathPageTitle">The real math behind every decision</h1>
        <p className="mathPageSub">
          A complete, step-by-step walkthrough of how Axiom-I analyzes an image or video and arrives at its final verdict. Every formula, every variable, and every value is explained below in the exact order the system processes them.
          {r && <> This analysis used <strong>{isVideo ? (r.analysis_mode === 'video_full' ? 'Video Physics' : 'Video Fallback') : isPhysics ? 'Full Physics' : 'Fallback'}</strong> mode
          and produced a verdict of <strong className={r.verdict === 'Fake' ? 'textFake' : 'textReal'}>{r.verdict}</strong> with {(r.confidence * 100).toFixed(1)}% confidence.</>}
        </p>
      </div>

      {!r && (
        <div className="emptyState">
           No analysis data found. Please go to the main page, analyze an image or video, and then click the "View full mathematics" button.
        </div>
      )}

      {renderVideoMath()}

      {/* STEP 1: Face Detection */}
      {r && isPhysics && (
        <div className="deepStep">
          <div className="deepStepHeader">
            <div className="deepStepNum">1</div>
            <span className="deepStepFile">face_detector.py</span>
            <h2 className="deepStepTitle">Face detection (MTCNN)</h2>
          </div>

          <p className="deepDesc">
            <strong>What it does:</strong> This file takes the uploaded image and searches for human faces in it. It uses a model called MTCNN, which scans the image and returns two things for each face it finds: the exact position of the face (as pixel coordinates), and a confidence score that tells how sure the model is that it found a real face.
          </p>
          <p className="deepDesc">
            <strong>Why it is needed:</strong> Fake images created by AI usually have their problems concentrated around the face area, such as wrong lighting, blending edges, or unnatural skin reflections. If we analyze the whole image including the background, these small signs of fakeness would get lost in all the background noise. So we first find the face and cut it out.
          </p>
          <p className="deepDesc">
            <strong>How it works:</strong> MTCNN gives us the face position as four numbers: [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the face. If multiple faces are found, the system picks the largest one. Then it adds 10% extra space around all sides of the face so that the jawline and forehead are also included in the crop.
          </p>

          {face && (
            <div className="derivBlock">
              <div className="derivLabel">Actual values from this analysis</div>
              <table className="varTable">
                <thead><tr><th>Variable</th><th>Value</th><th>Meaning</th></tr></thead>
                <tbody>
                  <tr><td className="mono">faces_detected</td><td className="mono">{r.faces_detected}</td><td>Total number of faces found in the image</td></tr>
                  <tr><td className="mono">bbox [x1, y1, x2, y2]</td><td className="mono">[{face.bbox.join(', ')}]</td><td>Pixel coordinates of the selected face</td></tr>
                  <tr><td className="mono">confidence</td><td className="mono">{(face.confidence * 100).toFixed(2)}%</td><td>How confident MTCNN is that this is a real face</td></tr>
                  <tr><td className="mono">w = x2 - x1</td><td className="mono">{face.bbox[2] - face.bbox[0]} px</td><td>Width of the face bounding box</td></tr>
                  <tr><td className="mono">h = y2 - y1</td><td className="mono">{face.bbox[3] - face.bbox[1]} px</td><td>Height of the face bounding box</td></tr>
                </tbody>
              </table>
            </div>
          )}

          <div className="derivBlock">
            <div className="derivLabel">Padding formula</div>
            <Tex math="\Delta x = 0.10 \times w, \quad \Delta y = 0.10 \times h" block />
            <Tex math="x_1' = \max(0,\; x_1 - \Delta x), \quad x_2' = \min(W_{\text{img}},\; x_2 + \Delta x)" block />
            {face && <>
              <p className="derivStep">Substituting actual values:</p>
              <p className="derivStep">
                <Tex math={`\\Delta x = 0.10 \\times ${face.bbox[2] - face.bbox[0]} = ${((face.bbox[2] - face.bbox[0]) * 0.1).toFixed(1)}`} /> pixels of padding horizontally.
              </p>
              <p className="derivStep">
                <Tex math={`\\Delta y = 0.10 \\times ${face.bbox[3] - face.bbox[1]} = ${((face.bbox[3] - face.bbox[1]) * 0.1).toFixed(1)}`} /> pixels of padding vertically.
              </p>
            </>}
          </div>
        </div>
      )}

      {/* STEP 2: Surface Normal Estimation */}
      {r && isPhysics && (
        <div className="deepStep">
          <div className="deepStepHeader">
            <div className="deepStepNum">2</div>
            <span className="deepStepFile">face_alignment.py</span>
            <h2 className="deepStepTitle">Surface normal estimation</h2>
          </div>

          <p className="deepDesc">
            <strong>What it does:</strong> A photograph is flat (2D), but a real face is curved (3D). This file tries to guess the 3D shape of the face from the flat image. It creates a "depth map" that shows which parts of the face are closer to the camera and which are farther away. Then it calculates a "normal vector" at every pixel, which is like a small arrow pointing straight outward from the skin surface.
          </p>
          <p className="deepDesc">
            <strong>Why it is needed:</strong> Later, the system needs to figure out how light reflects off the face. To do that, it needs to know the angle of the skin at every point. For example, the tip of the nose points forward (toward the camera), while the sides of the cheeks point sideways. These angles decide where shiny reflections appear.
          </p>

          <div className="derivBlock">
            <div className="derivLabel">Step 1: Depth estimation</div>
            <p className="derivStep">First, the face image is converted to grayscale (black and white). Then a bilateral filter is applied (with settings d=9, sigma_color=75, sigma_space=75). This filter smooths out the skin texture but keeps the sharp edges like the nose and jawline intact. The result is treated as a rough depth map, where brighter pixels mean that part of the face is closer to the camera.</p>
            <p className="derivStep">Since a face is roughly shaped like a sphere, a radial correction is added. It assumes the center of the face is closest to the camera, and the edges curve away. This correction is blended into the depth map:</p>
            <Tex math="r(x,y) = \sqrt{\left(\frac{x - c_x}{c_x}\right)^2 + \left(\frac{y - c_y}{c_y}\right)^2}" block />
            <Tex math="D(x,y) = 0.75 \times B(x,y) + 0.25 \times \max(0,\; 1 - r^2)" block />
            <p className="derivStep">Here, B(x,y) is the bilaterally filtered grayscale value (normalized to 0-1). The coefficient 0.75 gives most weight to the actual image data, while 0.25 adds the spherical curvature assumption.</p>
          </div>

          <div className="derivBlock">
            <div className="derivLabel">Step 2: Normal vector calculation</div>
            <p className="derivStep">The Sobel operator is applied to find the slopes (gradients) of the depth map. It tells us how quickly the depth changes in the horizontal (x) and vertical (y) directions. From these slopes, a raw direction vector is formed at each pixel:</p>
            <Tex math="\mathbf{V} = [-2 \times d_x,\; -2 \times d_y,\; 1]" block />
            <p className="derivStep">This vector is then scaled so its length becomes exactly 1. This is called "normalization" and it gives us the final unit normal vector:</p>
            <Tex math="\mathbf{N} = \frac{\mathbf{V}}{||\mathbf{V}||} = \frac{[-2d_x,\; -2d_y,\; 1]}{\sqrt{4d_x^2 + 4d_y^2 + 1}}" block />
            <p className="derivStep">The result is saved as a colorful "normal map" image. The red, green, and blue channels of this image store the x, y, and z components of the normal direction at each pixel.</p>
          </div>
        </div>
      )}

      {/* STEP 3: Retinex */}
      {r && isPhysics && (
        <div className="deepStep">
          <div className="deepStepHeader">
            <div className="deepStepNum">3</div>
            <span className="deepStepFile">retinex.py</span>
            <h2 className="deepStepTitle">Multi-Scale Retinex texture extraction</h2>
          </div>

          <p className="deepDesc">
            <strong>What it does:</strong> Any image you see is made up of two things combined together: the actual color and pattern of the skin (called texture or reflectance), and the light shining on it (called illumination). This file separates these two parts. It removes the effect of lighting and gives us a clean texture-only map of the face.
          </p>
          <p className="deepDesc">
            <strong>Why it is needed:</strong> In the next steps, we need the pure skin texture to compare it against the shiny reflections. If we skip this step, the lighting effects would mix with the texture and make the comparison inaccurate. The basic idea is simple: Image = Texture multiplied by Illumination.
          </p>

          <div className="derivBlock">
            <div className="derivLabel">Calculation</div>
            <p className="derivStep">Since Image = Texture x Illumination, we can take the logarithm of both sides to get: log(Image) = log(Texture) + log(Illumination). This converts multiplication into addition, which makes it easy to separate the two parts.</p>
            <p className="derivStep">The illumination is estimated by heavily blurring the image using a Gaussian filter. A heavily blurred image represents the smooth lighting component. We do this at three different blur sizes (sigma = 15, 80, and 120) to capture different levels of detail:</p>
            <Tex math="\text{Retinex}_\sigma = \log(I + 1) - \log(G_\sigma * I + 1)" block />
            <p className="derivStep">Sigma=15 captures tiny details like skin pores. Sigma=80 captures medium features. Sigma=120 captures large shadow areas. The final texture map is the average of all three results:</p>
            <Tex math="\text{MSR} = \frac{1}{3}\Big[\text{Retinex}_{15} + \text{Retinex}_{80} + \text{Retinex}_{120}\Big]" block />
          </div>
        </div>
      )}

      {/* STEP 4: Spherical Harmonics */}
      {r && isPhysics && (
        <div className="deepStep">
          <div className="deepStepHeader">
            <div className="deepStepNum">4</div>
            <span className="deepStepFile">illumination.py</span>
            <h2 className="deepStepTitle">Spherical Harmonics lighting decomposition</h2>
          </div>

          <p className="deepDesc">
            <strong>What it does:</strong> This file figures out exactly how the light is falling on the face. It uses 9 special mathematical functions called Spherical Harmonics to describe the lighting from all directions. Once the lighting is known, it splits the light into two parts: ambient light (the general background glow that comes from everywhere) and direct light (the focused light coming from a specific direction, like a lamp or the sun).
          </p>
          <p className="deepDesc">
            <strong>Why it is needed:</strong> In the next step, we need to create a fake "ideal" image of the face that shows only the matte (non-shiny) appearance. To do that, we first need to know the exact lighting. If we get the lighting wrong, the specular (shiny) residual will also be wrong, and the entire analysis will be inaccurate.
          </p>

          <div className="derivBlock">
            <div className="derivLabel">Calculation</div>
            <p className="derivStep">1. For every pixel in the face, the surface normal direction [nx, ny, nz] (from Step 2) is plugged into 9 fixed formulas called Spherical Harmonic basis functions. These give 9 numbers per pixel.</p>
            <p className="derivStep">2. All these 9-number rows are stacked into a big matrix called A. The target value y is computed by dividing the original image by the texture: y = Image / Texture. This isolates just the lighting component.</p>
            <p className="derivStep">3. The system solves for the 9 best lighting coefficients (called gamma) that explain the observed lighting. This is done separately for the Red, Green, and Blue channels using the following formula:</p>
            <Tex math="\gamma = (A^T A + \lambda I_{9 \times 9})^{-1} \times A^T \times y" block />
            <p className="derivStep">Here, lambda = 0.001 is a tiny number added to prevent division-by-zero errors during the calculation. The result is a 9x3 table of lighting coefficients (9 coefficients for each of the 3 color channels).</p>
            <p className="derivStep">4. The first coefficient (gamma_0) represents the ambient light, which is the overall background glow. The remaining 8 coefficients represent directional light from different angles:</p>
            <Tex math="\text{Ambient} = Y_0 \cdot \gamma_0, \quad \text{Direct} = \text{Total Illumination} - \text{Ambient}" block />
          </div>
        </div>
      )}

      {/* STEP 5: Specular Residual */}
      {r && isPhysics && (
        <div className="deepStep">
          <div className="deepStepHeader">
            <div className="deepStepNum">5</div>
            <span className="deepStepFile">specular.py</span>
            <h2 className="deepStepTitle">Specular residual extraction</h2>
          </div>

          <p className="deepDesc">
            <strong>What it does:</strong> This file creates an ideal version of the face that has no shiny reflections at all. It imagines what the face would look like if the skin were completely matte (like chalk). Then it subtracts this matte image from the real image. Whatever is left over is the shiny part, called the Specular Residual (SPR).
          </p>
          <p className="deepDesc">
            <strong>Why it is needed:</strong> In real photos, the shiny reflections on the skin (like on the nose tip or forehead) are created by real physics. They depend on the actual 3D shape and the real light direction. But in fake images made by AI, the model does not understand real physics. It often "draws" fake shiny spots directly into the skin texture. This creates a suspicious connection between the texture and the reflections that should not exist in real photos.
          </p>

          <div className="derivBlock">
            <div className="derivLabel">Calculation</div>
            <p className="derivStep">1. Convert the texture from log domain back to linear domain:</p>
            <Tex math="T_{\text{linear}} = e^{\text{MSR}}" block />
            <p className="derivStep">2. Compute the ideal matte (Lambertian) image by multiplying the total light by the linear texture:</p>
            <Tex math="\text{Lambertian} = (\text{Ambient} + \text{Direct}) \times T_{\text{linear}}" block />
            <p className="derivStep">3. Subtract the Lambertian prediction from the actual image. Any positive remainder is a specular highlight:</p>
            <Tex math="\text{SPR} = \max\Big(0,\; \frac{\text{Image}}{255} - \text{Lambertian}\Big)" block />
            <p className="derivStep">The SPR is clipped to [0, 3] to avoid extreme outliers.</p>
          </div>
        </div>
      )}

      {/* STEP 6: Specular Anomaly Score */}
      {r && isPhysics && (
        <div className="deepStep">
          <div className="deepStepHeader">
            <div className="deepStepNum">6</div>
            <span className="deepStepFile">sri_net.py</span>
            <h2 className="deepStepTitle">Specular anomaly score</h2>
          </div>

          <p className="deepDesc">
            <strong>What it does:</strong> This is where the system checks if the shiny reflections (SPR from Step 5) and the skin texture (from Step 3) look too similar to each other. In a real photo, these two things should be completely independent. Shiny spots are caused by light, not by skin patterns. But in a fake image, the AI often mixes them together, so they end up looking alike.
          </p>
          <p className="deepDesc">
            <strong>How the score works:</strong> A score close to 0 means the reflections and texture look different (normal, likely real). A score close to 1 means they look very similar (suspicious, likely fake).
          </p>

          <div className="derivBlock">
            <div className="derivLabel">Calculation</div>
            <p className="derivStep">1. Flatten both the texture map and the SPR map into 1D arrays.</p>
            <p className="derivStep">2. Compute the Normalized Cross-Correlation (NCC), which is equivalent to the Pearson correlation coefficient. It ranges from -1 (opposite) to +1 (identical):</p>
            <Tex math="\text{NCC} = \frac{\sum (T_i - \bar{T})(S_i - \bar{S})}{\sqrt{\sum (T_i - \bar{T})^2 \cdot \sum (S_i - \bar{S})^2}}" block />
            <p className="derivStep">3. Map the NCC to an anomaly probability using a sigmoid function. The center point is 0.30 (NCC above 0.30 is suspicious), and the steepness is 15:</p>
            <Tex math={`\\text{Anomaly} = \\sigma(15 \\cdot (\\text{NCC} - 0.30)) = \\frac{1}{1 + e^{-15(\\text{NCC} - 0.30)}} = ${f(df.specular)}`} block />
          </div>

          {df.specular !== undefined && (
            <div className="derivBlock">
              <div className="derivLabel">Actual values from this analysis</div>
              <table className="varTable">
                <thead><tr><th>Variable</th><th>Value</th></tr></thead>
                <tbody>
                  <tr><td className="mono">Specular anomaly score</td><td className="mono">{f(df.specular)}</td></tr>
                  <tr><td className="mono">Interpretation</td><td>{Number(df.specular) >= 0.6 ? 'High anomaly (likely fake specular pattern)' : Number(df.specular) <= 0.3 ? 'Low anomaly (specular pattern looks natural)' : 'Moderate anomaly'}</td></tr>
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* STEP 7: Frequency */}
      {r && (
        <div className="deepStep">
          <div className="deepStepHeader">
            <div className="deepStepNum">{isPhysics ? 7 : 1}</div>
            <span className="deepStepFile">frequency.py</span>
            <h2 className="deepStepTitle">Frequency domain analysis (FFT)</h2>
          </div>
          <p className="deepDesc"><strong>What it does:</strong> Every image is made up of patterns that repeat at different speeds. Smooth areas like the sky have "slow" patterns (low frequency), while sharp edges and tiny details have "fast" patterns (high frequency). This file converts the image from normal pixel form into a frequency map using a technique called the Fast Fourier Transform. It then measures how much energy is in the high-frequency area compared to the low-frequency area.</p>
          <p className="deepDesc"><strong>Why it is needed:</strong> AI image generators use a technique called "upsampling" (transposed convolutions) to create high-resolution images from small ones. This process leaves invisible repeating grid patterns in the image. These patterns are invisible to the human eye, but they show up clearly in the frequency map as unusual spikes in the high-frequency region.</p>
          <div className="derivBlock">
            <div className="derivLabel">Calculation</div>
            <p className="derivStep">1. The image is first converted to grayscale. Then a 2D FFT is applied, which converts the image from pixels into a map of frequencies. The zero-frequency (the average brightness) is moved to the center of the map.</p>
            <p className="derivStep">2. The power of each frequency is calculated by squaring its magnitude.</p>
            <p className="derivStep">3. The frequencies are organized into rings based on how far they are from the center. Frequencies near the center are low (smooth, gradual changes). Frequencies far from the center are high (sharp edges, fine details). The inner 15% of the radius is considered low-frequency, and the outer 30% is considered high-frequency.</p>
            <Tex math="\text{HFER} = \frac{\sum \text{Power in high-frequency ring}}{\sum \text{Power in low-frequency ring}}" block />
            <p className="derivStep">4. Map the log of HFER to an anomaly probability. The sigmoid has steepness 3.0 and center point at log10(HFER) = -4.5:</p>
            <Tex math={`\\text{Anomaly} = \\frac{1}{1 + e^{-3.0 \\times (\\log_{10}(\\text{HFER}) + 4.5)}} = ${f(isPhysics ? df.frequency : fb.frequency)}`} block />
          </div>
          {(df.frequency !== undefined || fb.frequency !== undefined) && (
            <div className="derivBlock">
              <div className="derivLabel">Actual values</div>
              <table className="varTable">
                <thead><tr><th>Variable</th><th>Value</th></tr></thead>
                <tbody>
                  <tr><td className="mono">Frequency anomaly</td><td className="mono">{f(isPhysics ? df.frequency : fb.frequency)}</td></tr>
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* STEP 8: Patch */}
      {r && isPhysics && (
        <div className="deepStep">
          <div className="deepStepHeader">
            <div className="deepStepNum">8</div>
            <span className="deepStepFile">patch_analysis.py</span>
            <h2 className="deepStepTitle">Patch noise consistency (PRNU)</h2>
          </div>
          <p className="deepDesc"><strong>What it does:</strong> This file divides the face into a 4x4 grid, creating 16 small patches. For each patch, it measures the level of noise. Then it checks whether the noise level is consistent (similar) across all 16 patches.</p>
          <p className="deepDesc"><strong>Why it is needed:</strong> Every real camera leaves a unique noise fingerprint on the images it takes. This noise is spread evenly across the entire image. But when an AI generates a face, or when someone pastes a fake face onto a real photo, different parts of the image may have very different noise levels. If the noise varies a lot from patch to patch, the image is likely manipulated.</p>
          <div className="derivBlock">
            <div className="derivLabel">Calculation</div>
            <p className="derivStep">1. The Laplacian operator (a type of edge detector) is applied to the grayscale image. This pulls out only the high-frequency noise, which is the camera sensor noise.</p>
            <p className="derivStep">2. The noise image is split into a 4x4 grid (16 patches). For each patch, the signal-to-noise ratio (SNR) is calculated. SNR tells us how "noisy" that patch is:</p>
            <Tex math="\text{SNR}_i = \frac{\text{std}(\text{patch}_i)}{\text{mean}(\text{patch}_i)}" block />
            <p className="derivStep">3. Now we check if all 16 SNR values are similar or very different. This is done using the Coefficient of Variation (CV), which is simply the standard deviation of the SNR values divided by their mean:</p>
            <Tex math="\text{CV} = \frac{\text{std}(\text{SNR}_1, \ldots, \text{SNR}_{16})}{\text{mean}(\text{SNR}_1, \ldots, \text{SNR}_{16})}" block />
            <p className="derivStep">4. A high CV means the noise levels are very different across the face, which is a sign of manipulation. The CV is converted to an anomaly score using a sigmoid with center 0.25:</p>
            <Tex math={`\\text{Anomaly} = \\frac{1}{1 + e^{20 \\times (\\text{CV} - 0.25)}} = ${f(df.patch_consistency)}`} block />
          </div>
          {df.patch_consistency !== undefined && (
            <div className="derivBlock">
              <div className="derivLabel">Actual values</div>
              <table className="varTable">
                <thead><tr><th>Variable</th><th>Value</th></tr></thead>
                <tbody><tr><td className="mono">Patch anomaly</td><td className="mono">{f(df.patch_consistency)}</td></tr></tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* STEP 9: Topology */}
      {r && isPhysics && (
        <div className="deepStep">
          <div className="deepStepHeader">
            <div className="deepStepNum">9</div>
            <span className="deepStepFile">topology.py</span>
            <h2 className="deepStepTitle">Topological complexity analysis</h2>
          </div>
          <p className="deepDesc"><strong>What it does:</strong> This file looks at the shape of the shiny reflections (the Specular Residual from Step 5). It counts how many separate bright spots exist and whether they have holes in them, at three different brightness levels.</p>
          <p className="deepDesc"><strong>Why it is needed:</strong> In real photos, shiny reflections on the skin are usually large, smooth, and continuous (like a single bright spot on the nose tip). But in AI-generated images, the reflections tend to be broken into many small, scattered pieces, like shattered glass. This fragmentation is a strong sign of fakeness.</p>
          <div className="derivBlock">
            <div className="derivLabel">Calculation</div>
            <p className="derivStep">1. Threshold the SPR magnitude at three levels: low (64), mid (128), high (192).</p>
            <p className="derivStep">2. At each level, count the number of connected components (C) and holes (H) using OpenCV.</p>
            <p className="derivStep">3. Compute weighted complexity. Higher thresholds and holes get more weight because they are rarer in real images:</p>
            <Tex math="C_{\text{total}} = 0.20 C_{\text{low}} + 0.35 C_{\text{mid}} + 0.45 C_{\text{high}} + 0.50 H_{\text{low}} + 0.80 H_{\text{mid}} + 1.10 H_{\text{high}}" block />
            <p className="derivStep">4. Map to anomaly probability with center 18 and steepness 0.11:</p>
            <Tex math={`\\text{Anomaly} = \\frac{1}{1 + e^{-0.11 \\times (C_{\\text{total}} - 18)}} = ${f(df.topology)}`} block />
          </div>
          {df.topology !== undefined && (
            <div className="derivBlock">
              <div className="derivLabel">Actual values</div>
              <table className="varTable">
                <thead><tr><th>Variable</th><th>Value</th></tr></thead>
                <tbody><tr><td className="mono">Topology anomaly</td><td className="mono">{f(df.topology)}</td></tr></tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* STEP 10: Wavelet */}
      {r && (
        <div className="deepStep">
          <div className="deepStepHeader">
            <div className="deepStepNum">{isPhysics ? 10 : 2}</div>
            <span className="deepStepFile">wavelet.py</span>
            <h2 className="deepStepTitle">Wavelet decomposition (DWT)</h2>
          </div>
          <p className="deepDesc"><strong>What it does:</strong> This file breaks the image into different directional components using a mathematical technique called Wavelet Transform. It separates the image into horizontal detail, vertical detail, and diagonal detail at multiple zoom levels.</p>
          <p className="deepDesc"><strong>Why it is needed:</strong> AI-generated faces often have unnatural artifacts along the jawline, hairline, or forehead edges due to the blending process. These artifacts show up most clearly in the diagonal detail band of the wavelet decomposition. Real photos have a natural balance of energy across all directions.</p>
          <div className="derivBlock">
            <div className="derivLabel">Calculation</div>
            <p className="derivStep">1. Convert to grayscale. Apply 2-level DWT to get detail sub-bands: LH (horizontal), HL (vertical), HH (diagonal).</p>
            <p className="derivStep">2. For each level, compute the average energy. The HH band gets 1.5x weight:</p>
            <Tex math="E_{\text{level}} = \frac{E_{LH} + E_{HL} + 1.5 \times E_{HH}}{3}" block />
            <p className="derivStep">3. Average the energy across all levels and map to anomaly probability:</p>
            <Tex math={`\\text{Anomaly} = \\frac{1}{1 + e^{-3 \\times (\\ln(1 + E_{\\text{avg}}) - 4.5)}} = ${f(isPhysics ? df.wavelet_score : fb.wavelet)}`} block />
          </div>
        </div>
      )}

      {/* STEP 11: ViT */}
      {r && (
        <div className="deepStep">
          <div className="deepStepHeader">
            <div className="deepStepNum">{isPhysics ? 11 : 3}</div>
            <span className="deepStepFile">vit_classifier.py</span>
            <h2 className="deepStepTitle">Vision Transformer classification (ViT)</h2>
          </div>
          <p className="deepDesc"><strong>What it does:</strong> This file uses a pre-trained deep learning model called a Vision Transformer (ViT). The model was specifically trained on thousands of real and fake images to learn the visual patterns that distinguish them.</p>
          <p className="deepDesc"><strong>Why it is needed:</strong> The previous steps (specular, frequency, patch, topology, wavelet) each look at one specific physical property. But there may be other subtle patterns of fakeness that these specific tests miss. The ViT model learns to detect any visual pattern, even ones that humans have not specifically designed a test for. It acts as a general-purpose safety net.</p>
          <div className="derivBlock">
            <div className="derivLabel">Calculation</div>
            <p className="derivStep">1. The image is preprocessed (resized and normalized) according to the model requirements.</p>
            <p className="derivStep">2. The model outputs raw logits for each class (Real and Fake).</p>
            <p className="derivStep">3. Softmax converts logits to probabilities. The probability of the "Fake" class is the ViT score:</p>
            <Tex math={`P(\\text{fake}) = \\text{softmax}(\\text{logits})[\\text{fake\\_id}] = ${f(isPhysics ? df.vit_score : fb.vit_score)}`} block />
          </div>
        </div>
      )}

      {/* STEP 12: Noisy-OR Fusion (Physics mode only) */}
      {r && isPhysics && sb && (
        <div className="deepStep">
          <div className="deepStepHeader">
            <div className="deepStepNum">12</div>
            <span className="deepStepFile">sri_net.py</span>
            <h2 className="deepStepTitle">Noisy-OR signal fusion</h2>
          </div>
          <p className="deepDesc"><strong>What it does:</strong> Now the system has 6 separate anomaly scores (specular, frequency, patch, topology, wavelet, and ViT). This file combines them all into one single number using a method called Noisy-OR.</p>
          <p className="deepDesc"><strong>Why this method instead of simple averaging:</strong> If we simply averaged all 6 scores, one very high score (say 0.95 from specular) could be dragged down by 5 low scores (say 0.1 each), making the average only about 0.24, which would wrongly classify the image as real. Noisy-OR avoids this problem. It works on the idea that if even one detector strongly says the image is fake, the final result should also be high.</p>
          <div className="derivBlock">
            <div className="derivLabel">Calculation with actual values</div>
            <p className="derivStep">Each signal has a fixed weight representing the system's trust in that detector:</p>
            <table className="varTable">
              <thead><tr><th>Signal</th><th>Weight</th><th>Score</th><th>P(Real) = 1 - w*s</th></tr></thead>
              <tbody>
                <tr><td className="mono">Specular</td><td className="mono">0.10</td><td className="mono">{f(df.specular)}</td><td className="mono">{f(1 - 0.10 * (df.specular || 0))}</td></tr>
                <tr><td className="mono">Frequency</td><td className="mono">0.18</td><td className="mono">{f(df.frequency)}</td><td className="mono">{f(1 - 0.18 * (df.frequency || 0))}</td></tr>
                <tr><td className="mono">Topology</td><td className="mono">0.22</td><td className="mono">{f(df.topology)}</td><td className="mono">{f(1 - 0.22 * (df.topology || 0))}</td></tr>
                <tr><td className="mono">Patch</td><td className="mono">0.22</td><td className="mono">{f(df.patch_consistency)}</td><td className="mono">{f(1 - 0.22 * (df.patch_consistency || 0))}</td></tr>
                <tr><td className="mono">Wavelet</td><td className="mono">0.15</td><td className="mono">{f(df.wavelet_score)}</td><td className="mono">{f(1 - 0.15 * (df.wavelet_score || 0))}</td></tr>
                <tr><td className="mono">ViT</td><td className="mono">0.13</td><td className="mono">{f(df.vit_score)}</td><td className="mono">{f(1 - 0.13 * (df.vit_score || 0))}</td></tr>
              </tbody>
            </table>
            <p className="derivStep">Multiply all P(Real) values together:</p>
            <Tex math={`P(\\text{All Real}) = ${[
              (1 - 0.10 * (df.specular || 0)),
              (1 - 0.18 * (df.frequency || 0)),
              (1 - 0.22 * (df.topology || 0)),
              (1 - 0.22 * (df.patch_consistency || 0)),
              (1 - 0.15 * (df.wavelet_score || 0)),
              (1 - 0.13 * (df.vit_score || 0)),
            ].map(v => v.toFixed(4)).join(' \\times ')} = ${f(1 - (sb.physics_ensemble || 0))}`} block />
            <Tex math={`\\text{Ensemble} = 1 - P(\\text{All Real}) = ${f(sb.physics_ensemble)}`} block />
            <p className="derivStep">The ensemble score is then passed through a sigmoid to produce the heuristic score:</p>
            <Tex math={`\\text{Heuristic} = \\sigma(8 \\times (${f(sb.physics_ensemble)} - 0.40)) = ${f(df.heuristic_score)}`} block />
          </div>
        </div>
      )}

      {/* Fallback: Weighted Combination */}
      {r && !isPhysics && !isVideo && (
        <div className="deepStep">
          <div className="deepStepHeader">
            <div className="deepStepNum">4</div>
            <span className="deepStepFile">pipeline.py</span>
            <h2 className="deepStepTitle">Weighted combination (Fallback mode)</h2>
          </div>
          <p className="deepDesc">Since no face was detected, the three signals are combined with fixed weights instead of Noisy-OR fusion:</p>
          <div className="derivBlock">
            <div className="derivLabel">Calculation with actual values</div>
            <Tex math={`\\text{Combined} = 0.30 \\times ${f(fb.frequency)} + 0.30 \\times ${f(fb.wavelet)} + 0.40 \\times ${f(fb.vit_score)} = ${f(fb.combined)}`} block />
            <Tex math={`\\text{Heuristic} = \\sigma(8 \\times (${f(fb.combined)} - 0.45)) = ${f(fb.calibrated)}`} block />
          </div>
        </div>
      )}

      {/* STEP 13: Calibration */}
      {r && !isVideo && (
        <div className="deepStep">
          <div className="deepStepHeader">
            <div className="deepStepNum">{isPhysics ? 13 : 5}</div>
            <span className="deepStepFile">evolution.py</span>
            <h2 className="deepStepTitle">Calibration model blending</h2>
          </div>
          <p className="deepDesc"><strong>What it does:</strong> The system does not rely only on the physics-based score. It also has a machine learning model that has been trained on user feedback (when users tell the system whether its prediction was right or wrong). This file blends the physics score with the learned score. The more feedback the system has received, the more it trusts the learned model.</p>
          <div className="derivBlock">
            <div className="derivLabel">Calculation with actual values</div>
            <p className="derivStep">The weight given to the learned model depends on how many feedback records are available. If there are fewer than 15 feedback records, the weight is set to 0, meaning only the physics-based heuristic is used. Once the model has enough data to prove itself (15 or more feedback records), the weight starts at 0.10 and gradually increases up to 0.80:</p>
            <Tex math="w = \begin{cases} 0 & \text{if feedback} < 15 \\ \text{clamp}(0.10 + 0.02 \times (n - 15),\; 0,\; 0.80) & \text{otherwise} \end{cases}" block />
            <p className="derivStep">The final score is then calculated as a blend of the two scores:</p>
            <Tex math={`\\text{Final} = w \\times \\text{Learned} + (1 - w) \\times \\text{Heuristic}`} block />
            {cb.model_weight !== undefined && (
              <>
                <p className="derivStep">Putting in the actual values from this analysis:</p>
                <Tex math={`\\text{Final} = ${f(cb.model_weight)} \\times ${f(cb.learned_score)} + ${f(typeof cb.model_weight === 'number' ? 1 - cb.model_weight : undefined)} \\times ${f(cb.heuristic_score)} = ${f(df.final_score)}`} block />
              </>
            )}
          </div>
        </div>
      )}

      {/* STEP 14: Final Verdict */}
      {r && !isVideo && (
        <div className="deepStep">
          <div className="deepStepHeader">
            <div className="deepStepNum">{isPhysics ? 14 : 6}</div>
            <span className="deepStepFile">pipeline.py</span>
            <h2 className="deepStepTitle">Final verdict</h2>
          </div>
          <div className="derivBlock">
            <div className="derivLabel">Decision rule</div>
            <p className="derivStep">The rule is simple: if the final score is 0.50 or higher, the image is classified as Fake. If it is below 0.50, it is classified as Real. The threshold 0.50 is the exact midpoint between 0 and 1.</p>
            <Tex math="\text{Verdict} = \begin{cases} \text{Fake} & \text{if Final Score} \geq 0.50 \\ \text{Real} & \text{if Final Score} < 0.50 \end{cases}" block />
            <p className="derivStep">The confidence tells us how sure the system is about its decision. It is calculated by measuring how far the score is from the 0.50 boundary. A score of 0.50 means zero confidence (could go either way). A score of 0.00 or 1.00 means 100% confidence.</p>
            <Tex math={`\\text{Confidence} = |\\text{Final Score} - 0.50| \\times 2 = |${f(df.final_score)} - 0.50| \\times 2 = ${(r.confidence * 100).toFixed(1)}\\%`} block />
          </div>
          <div className="derivBlock">
            <div className="derivLabel">Final result</div>
            <table className="varTable">
              <thead><tr><th>Output</th><th>Value</th></tr></thead>
              <tbody>
                <tr><td className="mono">Final score</td><td className="mono">{f(df.final_score)}</td></tr>
                <tr><td className="mono">Verdict</td><td className={`mono ${r.verdict === 'Fake' ? 'textFake' : 'textReal'}`}>{r.verdict}</td></tr>
                <tr><td className="mono">Confidence</td><td className="mono">{(r.confidence * 100).toFixed(1)}%</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

    </div>
  );
}
