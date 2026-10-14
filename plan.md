# Full plan: AI Deepfake, Voice Clone & Synthetic Media Detection (Indux #9)

I don't know your Indux deadline. The timeline below assumes about 4 weeks, so check the Schedule page and compress or stretch it to fit.

## 1. Goal and scope

**Problem:** Help users check whether an image, video or voice recording is authentic or AI-generated, with measurable accuracy and a clear explanation.

**Project name idea:** *TruthLens* (or any name you like).

**MVP (must ship):**
- Image and video deepfake detection with a heatmap explanation
- Voice-clone detection for audio clips
- Fusion into a single verdict and confidence score
- Web app with upload, results page and a downloadable report
- Evaluation with standard metrics, including a cross-dataset test

**Stretch (only if time allows):**
- Text/NLP branch (AI-generated captions or transcripts, using DistilBERT + SHAP)
- Audio-visual lip-sync check
- Batch upload and a history dashboard
- A browser extension or API key system

Build the MVP first. A polished MVP beats an unfinished full system.

## 2. System architecture

```
User upload
   ↓
Input router (detect type: image / video / audio / text)
   ↓
Preprocessing
   ├─ Video/Image → frame sampling → face detection → crops
   ├─ Audio → resample 16 kHz → trim → log-mel spectrogram
   └─ Text → clean → tokenise
   ↓
Detection branches (run in parallel)
   ├─ Visual model → score + Grad-CAM
   ├─ Audio model → score + saliency map
   └─ Text model → score + SHAP tokens
   ↓
Fusion layer (weighted → learned)
   ↓
Verdict (Real / Likely synthetic / Uncertain) + confidence
   ↓
Explanation panel + PDF report
```

**Design rule:** Every branch returns the same output shape: `{score, confidence, evidence}`. This keeps fusion and the UI simple, and lets you add or drop a branch without rewriting anything.

**Three verdicts instead of two:** Include "Uncertain" for scores near 0.5. It is more honest and judges notice it.

## 3. Data plan

| Branch | Datasets | Notes |
|---|---|---|
| Visual | FaceForensics++, Celeb-DF v2, a subset of DFDC | Request access in Week 1. Use subsets of a few thousand frames, not the full sets. |
| Audio | ASVspoof 2019 LA, WaveFake | ASVspoof gives you EER, the standard benchmark metric. |
| Audio-visual (stretch) | FakeAVCeleb | Contains both fake video and fake audio, useful for the fusion test. |
| Text (stretch) | Public human vs AI text datasets | Reuse your Tactic Detector pipeline. |

**Splits:** Train/validation/test split by **identity or source video**, never by frame. Frames from the same video in both train and test inflates accuracy badly.

**Custom test set:** Create 50–100 samples yourselves (clips you record, plus fakes from free tools) to test on "unseen" media. It makes a strong demo.

**Ethics:** Use only datasets under their license terms, don't store users' uploads longer than needed, and say so in the app.

## 4. Models in detail

### Visual branch
- **Face detection:** MediaPipe or MTCNN, sample 1 frame per 0.5 s or 16–32 frames per video.
- **Backbone:** EfficientNet-B0 or MobileNetV2 (you already know MobileNetV2), pretrained on ImageNet, fine-tuned for real/fake.
- **Augmentation:** Compression (JPEG, H.264), blur, brightness, flip. This matters for real-world robustness.
- **Video score:** Average (or attention-weight) the frame scores.
- **Explainability:** Grad-CAM overlay on the face.
- **Option:** Add a frequency-domain feature (FFT) channel, since many generators leave spectral artifacts.

### Audio branch
- **Features:** Log-mel spectrogram (or LFCC) at 16 kHz, 4-second windows.
- **Model:** Small CNN, or wav2vec2 embeddings with a light classifier head. Start with the CNN and upgrade only if results are weak.
- **Metric:** EER and AUC.
- **Explainability:** Saliency or Grad-CAM on the spectrogram, showing which time-frequency regions looked synthetic.

### Text branch (stretch)
- DistilBERT fine-tuned on human vs AI text, with SHAP token highlights.

### Fusion
1. **Version 1:** Weighted average of available branch scores (weights tuned on validation data).
2. **Version 2:** Logistic regression or XGBoost on `[visual_score, audio_score, text_score, agreement_flag]`.
3. **Missing branches:** If the input has no audio, skip that branch and report which modalities were used.
4. **Calibration:** Apply temperature scaling or Platt scaling so a "90%" really means about 90%.

### Lip-sync check (stretch)
Compare mouth movement with the audio. A mismatch is a strong cue for voice-over fakes. If it is too complex, mention it as future work.

## 5. Evaluation plan

| Test | Metric | Why |
|---|---|---|
| Within-dataset test | Accuracy, F1, AUC | Baseline |
| **Cross-dataset** (train FF++, test Celeb-DF) | AUC | Shows real generalization |
| Compression robustness | AUC at several quality levels | Social media recompresses everything |
| Audio spoof detection | EER, AUC | Standard benchmark |
| Ablation | Visual only vs audio only vs fused | Proves fusion helps |
| Explanation sanity check | Do heatmaps hit the face/manipulated region? | Builds trust |
| Latency | Seconds per image, per 10 s video | Practicality |

Report failures honestly. A table showing where the model breaks is more credible than 99% on a single dataset.

## 6. Application design

**Backend (FastAPI):**
- `POST /analyze` takes a file and returns JSON with the verdict, scores, and base64 heatmaps.
- `GET /report/{id}` returns the PDF.
- `GET /health` for monitoring.
- File limits: size and duration caps, allowed formats, temporary storage with auto-delete.

**Frontend:**
- Drag-and-drop upload
- Results page with a verdict badge (green/amber/red), confidence gauge, per-modality scores, heatmap overlay with a slider, and a plain-language explanation ("Unnatural texture around the mouth; audio shows synthetic spectral patterns")
- Download report button
- Disclaimer: "Automated detection is probabilistic, not proof"

**Report (PDF):** Verdict, scores, heatmaps, models used, timestamp, and file hash.

## 7. Tech stack

- **ML:** Python 3.12, PyTorch, torchvision, timm, librosa, torchaudio, Hugging Face Transformers, OpenCV, MediaPipe, SHAP, scikit-learn/XGBoost
- **Backend:** FastAPI, Uvicorn, Docker
- **Frontend:** React (or plain HTML/JS if the team is small)
- **Training:** Kaggle or Colab GPUs
- **Deployment:** Docker image on Hugging Face Spaces or Google Cloud Run, frontend on Vercel
- **Tracking:** Git/GitHub, Weights & Biases or simple CSV logs

## 8. Repository structure

```
truthlens/
├── data/                # scripts only, not raw data
├── notebooks/           # experiments
├── src/
│   ├── preprocessing/
│   ├── models/          # visual.py, audio.py, text.py, fusion.py
│   ├── explain/         # gradcam.py, saliency.py, shap_text.py
│   ├── training/
│   └── evaluation/
├── api/                 # FastAPI app
├── web/                 # frontend
├── reports/             # figures, metrics tables
├── docker/
├── README.md
└── requirements.txt
```

## 9. Week-by-week timeline

### Week 1: Setup and baselines
- Finalize roles, create the repo and set up Colab/Kaggle
- Request dataset access (do this on Day 1)
- Build preprocessing: frame sampling, face crops, audio spectrograms
- Train a quick baseline in each branch
- **Deliverable:** Baseline scores and a working preprocessing pipeline

### Week 2: Core training
- Fine-tune the visual model with augmentation
- Train the audio model and compute EER
- Start the text branch if you have capacity
- Set up the evaluation scripts
- **Deliverable:** Trained visual and audio models with metrics tables

### Week 3: Fusion and explainability
- Build the fusion layer and calibration
- Add Grad-CAM, spectrogram saliency and SHAP
- Run cross-dataset, compression and ablation tests
- Start the FastAPI backend
- **Deliverable:** End-to-end pipeline working from the command line

### Week 4: App, polish, demo
- Build the frontend and PDF report
- Dockerize and deploy
- Test with your custom unseen samples
- Write the README, record a backup demo video, prepare the slides
- **Deliverable:** Live demo, repository, report and presentation

**Buffer:** Reserve the last 2–3 days purely for fixes and rehearsal. Do not start new features in that window.

## 10. Team roles

| Role | Responsibilities |
|---|---|
| **Visual & fusion lead (you)** | Visual model, fusion, overall architecture |
| **Audio lead** | Audio dataset, model, EER evaluation |
| **App lead** | FastAPI, frontend, deployment, PDF report |
| **Evaluation & docs lead** | Metrics, ablations, cross-dataset tests, slides, README |

If you have fewer people, merge app and evaluation, and drop the text and lip-sync stretch goals.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Dataset access delayed | Apply immediately, and start with openly downloadable subsets |
| Poor generalization to new fakes | Cross-dataset training, strong augmentation, honest reporting |
| GPU limits | Small backbones, subsets, mixed precision, Colab/Kaggle |
| Audio branch underperforms | Use pretrained wav2vec2 features, or reduce audio to a smaller module |
| Heavy video processing is slow | Sample fewer frames and cap clip length at 10–15 s |
| Deployment issues | Test the Docker build in Week 3, not Week 4 |
| Overclaiming accuracy | Always show cross-dataset results and the "Uncertain" category |

## 12. Final deliverables for Indux

1. Working web app (live link plus a backup demo video)
2. GitHub repository with a clear README
3. Short technical report: problem, method, results table, limitations
4. Presentation slides
5. Metrics sheet: accuracy, F1, AUC, EER, cross-dataset and ablation results

## 13. Demo script (3–4 minutes)

1. State the problem: deepfakes are spreading, and people can't verify media.
2. Upload a real video and show the green verdict with a clean heatmap.
3. Upload a deepfake video and show the red verdict, the heatmap on the manipulated region, and the reason.
4. Upload a cloned-voice clip and show the audio saliency.
5. Show the results table, including where it fails.
6. Close with limitations and future work (new generators, real-time detection, a browser extension).

## 14. First 48 hours checklist

- [ ] Confirm the team and roles
- [ ] Submit dataset access requests (FaceForensics++, ASVspoof, Celeb-DF)
- [ ] Create the GitHub repo with the structure above
- [ ] Set up a shared Colab/Kaggle workspace
- [ ] Build a 100-image, 50-clip mini dataset for quick testing
- [ ] Decide the final project name and write a one-paragraph proposal

I can put this in a Word or PDF file if you'd like to keep it. I can also write the starter FastAPI backend and visual-branch training code, or draft the one-page proposal for submission.