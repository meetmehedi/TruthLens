# TruthLens: Multimodal Synthetic Media & Voice Clone Detection System

**Hackathon Track / Problem Statement:** Indux #9 – AI Deepfake, Voice Clone & Synthetic Media Detection  
**Team / Project Name:** TruthLens  
**Target Event:** Vikrant University Hackathon  

---

## 1. Executive Summary & Problem Statement

Generative AI tools (diffusion models, GANs, neural voice synthesizers, real-time face-swap pipelines) now produce media indistinguishable from authentic recordings to human observers. This creates severe vulnerabilities:
- **Financial & Social Engineering Fraud:** Executive audio impersonation, CEO fraud, unauthorized voice-cloned authorization.
- **Disinformation & Reputation Harm:** Weaponized deepfake videos of public figures, political disinformation, and non-consensual synthetic media.
- **Black-Box Skepticism:** Existing automated detectors often fail in practical use because they output opaque binary labels ("Fake: 98%") without verifiable visual or acoustic evidence, and break down under common social-media recompression (JPEG/H.264).

**TruthLens** is an end-to-end, explainable synthetic media forensic platform. It inspects images, video streams, and audio clips using deep convolutional and spectral feature extraction, combines them through a calibrated late-fusion layer, outputs a **tri-state verdict** (*Authentic*, *Likely Synthetic*, or *Uncertain*), and provides visual **Grad-CAM heatmaps** and **acoustic spectrogram saliency maps** alongside an exportable cryptographic forensic PDF report.

---

## 2. Technical Innovation & System Architecture

```
                                [ Incoming Media Upload ]
                                            │
                                  [ Input Media Router ]
                       ┌────────────────────┼────────────────────┐
                       ▼                    ▼                    ▼
               [ Image / Video ]         [ Audio ]       [ Text / Metadata ]
                       │                    │                    │
              Frame Extraction &     16 kHz Resample &    Tokenization &
             MediaPipe Face Cropping  Log-Mel Spectrogram   NLP Analysis
                       │                    │                    │
                       ▼                    ▼                    ▼
               [ Visual Backbone ]   [ Audio Backbone ]   [ Text Classifier ]
               EfficientNet-B0/      Small Spectrogram     DistilBERT Head
               MobileNetV2 (FFT)      CNN / wav2vec2            (SHAP)
                       │                    │                    │
                       ▼                    ▼                    ▼
                Grad-CAM Heatmap     Acoustic Saliency      SHAP Tokens
                       └────────────────────┬────────────────────┘
                                            ▼
                           [ Calibrated Late-Fusion Layer ]
                         Weighted Ensemble + Platt Scaling
                                            │
                                            ▼
                      [ Tri-State Verdict & Confidence Score ]
                        • Authentic  • Uncertain  • Synthetic
                                            │
                                            ▼
                     [ Explainable Web Dashboard & PDF Report ]
```

### Core Differentiators

1. **Multimodal Late-Fusion with Modality Agnosticism:**  
   Every analytical pipeline adheres to a unified interface `{modality, score, confidence, evidence_tensor}`. If an uploaded video has no audio track, or an audio clip has no visual track, the fusion engine dynamically reweights without pipeline failure.
2. **Transparent, Interpretable Forensics (XAI):**  
   Rather than treating the neural network as an oracle, TruthLens generates pixel-level Grad-CAM overlays highlighting facial boundary discrepancies, warping artifacts, and frequency-domain anomalies, alongside acoustic spectrogram saliency maps.
3. **Calibrated Tri-State Verdicts:**  
   Instead of forcing borderline samples ($0.45 \le p \le 0.55$) into binary false alarms, TruthLens designates them as **"Uncertain"**, drastically reducing costly false accusations and fostering trust among reviewers and judges.
4. **Compression Robustness Evaluation:**  
   Evaluated not only in-distribution, but explicitly benchmarked against cross-dataset validation (trained on FaceForensics++, tested on Celeb-DF v2) and degraded quality levels (JPEG/H.264 compression).

---

## 3. Technology Stack & Implementation Plan

| Component | Technologies & Frameworks |
|---|---|
| **Deep Learning & Forensics** | PyTorch 2.x, `torchvision`, `torchaudio`, `timm`, Hugging Face Transformers |
| **Preprocessing & Vision** | MediaPipe (Face Mesh/Detection), OpenCV (`cv2`), Librosa, SciPy (FFT) |
| **Explainability (XAI)** | PyTorch-Grad-CAM, Saliency Mapping, SHAP |
| **Backend API** | FastAPI, Uvicorn, Pydantic, Python-Multipart, ReportLab / WeasyPrint (PDF) |
| **Frontend UI** | Modern Vanilla Web Stack (HTML5/CSS3/ES6+), Interactive Heatmap Slider, Canvas |
| **Evaluation & Datasets** | FaceForensics++ (c23/c40), Celeb-DF v2, ASVspoof 2019 LA, Custom In-the-Wild Set |

---

## 4. Expected Deliverables & Impact

1. **Production-Ready Web Application:** Intuitive drag-and-drop media analyzer with instant tri-state results, confidence dials, interactive heatmap slider overlays, and one-click forensic PDF download.
2. **Standardized REST API:** High-throughput `POST /analyze` and `GET /report/{id}` endpoints suitable for enterprise integration, content moderation queues, or browser extensions.
3. **Rigorous Benchmark Report:** Audited accuracy, F1-score, AUC-ROC, Equal Error Rate (EER), and cross-dataset generalization metrics with open disclosure of edge cases and limitations.

---

# Team Action Plan: First 48-Hour Execution Checklist

To maximize velocity and ensure all components integrate smoothly ahead of demo day, execute this checklist across the first 48 hours:

### Phase 1: Hours 0–12 (Foundation & Setup)
- [ ] **Role Allocation:**
  - **Member 1 (Visual & Fusion Lead):** Face crop pipeline, visual backbone training/inference, Grad-CAM generation.
  - **Member 2 (Audio Lead):** Audio extraction, 16 kHz resampler, log-mel spectrogram pipeline, CNN classifier.
  - **Member 3 (App & API Lead):** FastAPI backend scaffold, multipart file upload handling, web UI dashboard.
  - **Member 4 (Data & Evaluation Lead):** Benchmark dataset collection, baseline evaluation scripts, PDF report generator.
- [ ] **Repository Scaffolding:** Set up Git repo with defined module structure (`truthlens/src/`, `truthlens/api/`, `truthlens/web/`).
- [ ] **Environment Setup:** Create shared Python virtual environment and push pinned `requirements.txt`.
- [ ] **Dataset Access & Mini-Benchmarking:**
  - Download accessible subsets of FaceForensics++ (or public deepfake samples) and ASVspoof 2019 / WaveFake.
  - Assemble a **Mini-Validation Set (50 images, 25 videos, 25 audio clips)** for local testing.

### Phase 2: Hours 12–24 (Pipeline Skeleton & Contracts)
- [ ] **Define Unified API Contract:** Lock the JSON schema returned by each modality analyzer:
  ```json
  {
    "modality": "visual",
    "score": 0.88,
    "verdict": "synthetic",
    "confidence": 0.92,
    "explanation": "High-frequency boundary artifacts detected around periocular/mouth regions",
    "heatmap_base64": "..."
  }
  ```
- [ ] **Visual Baseline:** Implement MediaPipe face detection + frame sampling (1 frame per 0.5s), feed into MobileNetV2/EfficientNet-B0 feature extractor.
- [ ] **Audio Baseline:** Implement audio extraction from video container via `ffmpeg` / `torchaudio`, generate $128 \times 128$ log-mel spectrogram.
- [ ] **FastAPI Skeleton:** Implement working `/health` and mock `/analyze` endpoint receiving media files.

### Phase 3: Hours 24–36 (Explainability & UI Integration)
- [ ] **Grad-CAM Integration:** Hook Grad-CAM to the final convolutional layer of the visual backbone; output blended heatmap on cropped face.
- [ ] **Spectrogram Saliency:** Compute gradient saliency over the log-mel spectrogram highlighting synthetic frequencies.
- [ ] **Frontend Upload & Display:** Wire the frontend to call `POST /analyze`, display the tri-state verdict pill (Green = Authentic, Amber = Uncertain, Red = Synthetic), confidence gauges, and split-screen comparison slider.

### Phase 4: Hours 36–48 (Fusion, Testing & Report Generation)
- [ ] **Implement Late Fusion Engine:** Calibrate modality combination with weighted scoring and fallback logic for missing channels.
- [ ] **PDF Forensic Report Generator:** Implement PDF builder containing timestamp, SHA-256 file hash, individual modality scores, and embedded heatmaps.
- [ ] **End-to-End Ingestion Run:** Verify full pipeline execution from web upload $\rightarrow$ inference $\rightarrow$ heatmap rendering $\rightarrow$ PDF generation in $< 3.5$ seconds for images and $< 10$ seconds for 10-second videos.
- [ ] **Internal Checkpoint:** Team review of live local demo; record initial backup demo screencast.
