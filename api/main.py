from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import os
import shutil
import hashlib
import time
import uuid
import tempfile
from typing import Optional

# Import TruthLens core engine
from src.preprocessing.face_extractor import FaceExtractor
from src.preprocessing.audio_processor import AudioProcessor
from src.models.visual import VisualDetector
from src.models.audio import AudioDetector
from src.models.text import TextDetector
from src.models.fusion import MultimodalFusionEngine

app = FastAPI(
    title="TruthLens API",
    description="Explainable Multimodal Synthetic Media & Voice Clone Detection Platform",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
face_extractor = FaceExtractor()
audio_processor = AudioProcessor()
visual_detector = VisualDetector()
audio_detector = AudioDetector()
text_detector = TextDetector()
fusion_engine = MultimodalFusionEngine()

# Web Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, "web")
TEMP_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

# Mount web assets if directory exists
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>TruthLens Backend Active</h1>")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "TruthLens Multimodal Forensic Engine",
        "models": {
            "visual": "MobileNetV2 + Spatial 2D-FFT Spectral ResNet",
            "audio": "16kHz STFT Log-Mel Acoustic Vocoder Analyzer",
            "text": "Lexical Burstiness & AI Phrase Marker NLP Head",
            "fusion": "Calibrated Dynamic Weight Late-Fusion Engine"
        },
        "device": str(visual_detector.device),
        "timestamp": time.time()
    }

@app.post("/api/analyze")
async def analyze_media(
    file: Optional[UploadFile] = File(None),
    text_content: Optional[str] = Form(None)
):
    """
    Main analysis endpoint. Accepts an image, video, or audio file,
    plus optional transcript text. Returns unified tri-state verdict,
    confidence metrics, and explainable visualizations.
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())[:8]

    if file is None and (not text_content or not text_content.strip()):
        raise HTTPException(status_code=400, detail="Please upload a media file or provide text.")

    file_hash = "N/A"
    filename = "text_payload"
    file_size_bytes = 0
    content_type = "text/plain"
    temp_path = None

    if file:
        filename = file.filename
        content_type = file.content_type or ""
        ext = os.path.splitext(filename)[1].lower()
        temp_path = os.path.join(TEMP_DIR, f"{analysis_id}_{filename}")
        
        # Save file and calculate SHA-256 for cryptographic chain of custody
        hasher = hashlib.sha256()
        with open(temp_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                hasher.update(chunk)
                buffer.write(chunk)
                file_size_bytes += len(chunk)
        file_hash = hasher.hexdigest()

    visual_result = None
    audio_result = None
    text_result = None

    try:
        # Determine media processing branches
        is_image = any(ext_match in content_type for ext_match in ["image", "jpeg", "png", "webp"]) or (temp_path and temp_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
        is_video = any(ext_match in content_type for ext_match in ["video", "mp4", "webm", "quicktime"]) or (temp_path and temp_path.lower().endswith(('.mp4', '.avi', '.mov', '.webm')))
        is_audio = any(ext_match in content_type for ext_match in ["audio", "wav", "mpeg", "ogg", "mp3"]) or (temp_path and temp_path.lower().endswith(('.wav', '.mp3', '.ogg', '.m4a', '.flac')))

        # Visual Branch (Image / Video)
        if temp_path and (is_image or is_video):
            if is_image:
                face_data = face_extractor.extract_from_image(temp_path)
                visual_result = visual_detector.predict(face_data, filename=filename)
            elif is_video:
                sampled = face_extractor.extract_from_video(temp_path, max_frames=8)
                if sampled:
                    # Score first high-confidence face frame
                    first_with_face = next((f for f in sampled if f["face_found"]), sampled[0])
                    visual_result = visual_detector.predict(first_with_face, filename=filename)
                    # Average scores over all sampled frames
                    all_scores = [visual_detector.analyze_crop(f["crop_rgb"], f["crop_bgr"], filename=filename)["score"] for f in sampled]
                    visual_result["video_average_score"] = round(float(sum(all_scores) / len(all_scores)), 4)
                    visual_result["frames_sampled"] = len(sampled)

        # Audio Branch (Audio files or Video with Audio)
        if temp_path and (is_audio or is_video):
            try:
                audio_payload = audio_processor.process(temp_path)
                if audio_payload.get("raw_signal_length", 0) > 0:
                    audio_result = audio_detector.predict(audio_payload, filename=filename)
            except Exception as e:
                # Silent failure if video has no audio track
                pass

        # Text Branch
        if text_content and text_content.strip():
            text_result = text_detector.predict(text_content.strip())

        # Multimodal Late Fusion
        fusion_verdict = fusion_engine.fuse(
            visual_result=visual_result,
            audio_result=audio_result,
            text_result=text_result
        )

        latency_sec = round(time.time() - start_time, 2)

        return {
            "status": "success",
            "analysis_id": analysis_id,
            "filename": filename,
            "file_hash_sha256": file_hash,
            "file_size_kb": round(file_size_bytes / 1024, 1),
            "latency_seconds": latency_sec,
            "verdict": fusion_verdict["verdict"],
            "badge_color": fusion_verdict["badge_color"],
            "synthetic_probability_percent": fusion_verdict["synthetic_probability_percent"],
            "confidence_percent": fusion_verdict["confidence_percent"],
            "summary": fusion_verdict["summary"],
            "reasons": fusion_verdict["reasons"],
            "modalities_analyzed": fusion_verdict["modalities_analyzed"],
            "normalized_weights": fusion_verdict["normalized_weights"],
            "visual_branch": visual_result,
            "audio_branch": audio_result,
            "text_branch": text_result
        }

    finally:
        # Clean up temporary uploaded file to preserve disk space and ensure user privacy
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.get("/api/demo-samples")
async def get_demo_samples():
    """Returns curated synthetic vs authentic sample metadata for instant testing."""
    return {
        "samples": [
            {
                "id": "deepfake_portrait",
                "label": "AI Face-Swap / GAN Portrait",
                "type": "image",
                "expected_verdict": "Likely Synthetic",
                "source": "Synthesized with StyleGAN / FaceForensics++ c23",
                "description": "Visual artifacts around hair-forehead boundary, spectral checkerboard roll-off."
            },
            {
                "id": "authentic_portrait",
                "label": "Authentic Photojournalist Capture",
                "type": "image",
                "expected_verdict": "Authentic",
                "source": "Uncompressed camera sensor",
                "description": "Natural skin porosity, organic eye reflection, coherent chromatic dispersion."
            },
            {
                "id": "cloned_voice",
                "label": "Cloned CEO Audio Recording",
                "type": "audio",
                "expected_verdict": "Likely Synthetic",
                "source": "Zero-shot voice clone via neural vocoder",
                "description": "Rigid harmonic resonance, 8kHz energy cutoff, lack of micro-tremor."
            }
        ]
    }
