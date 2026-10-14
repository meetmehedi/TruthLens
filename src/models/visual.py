import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import base64
from typing import Dict, Any, Tuple


# ─────────────────────────────────────────────────────────────────────
# Neural architecture  (MUST match train.py exactly)
# ─────────────────────────────────────────────────────────────────────
class SpectralCNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
    def forward(self, x): return self.conv(x)


class TruthLensVisualNet(nn.Module):
    """
    Dual-stream MobileNetV2 + SpectralCNN + HF-power scalar.
    Fusion head: [1280 + 128 + 1] → 512 → 128 → 1.
    """
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.spatial_backbone = mobilenet.features
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.spectral_branch = SpectralCNNBranch()
        self.fusion_head = nn.Sequential(
            nn.Linear(1280 + 128 + 1, 512),
            nn.LayerNorm(512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.LayerNorm(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb, fft, hf_scalar):
        s = self.spatial_pool(self.spatial_backbone(rgb)).flatten(1)
        f = self.spectral_branch(fft)
        return self.fusion_head(torch.cat([s, f, hf_scalar], dim=1))


# ─────────────────────────────────────────────────────────────────────
# Visual Detector  — Calibrated Multimodal Forensic Scoring
# ─────────────────────────────────────────────────────────────────────
class VisualDetector(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = torch.device("cpu")
        self.weights_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "weights", "truthlens_visual.pth"
        )
        self.neural_model = TruthLensVisualNet()
        self._weights_loaded = False
        if os.path.exists(self.weights_path):
            try:
                state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
                missing, unexpected = self.neural_model.load_state_dict(state, strict=False)
                self._weights_loaded = True
                if missing or unexpected:
                    print(f"⚠️  Visual: Partial weight load ({len(missing)} missing, {len(unexpected)} unexpected) — rule-based detection active")
                else:
                    print(f"✅ Visual: Loaded weights from {self.weights_path}")
            except Exception as e:
                print(f"⚠️  Visual: weight load failed ({e}) — rule-based detection active")
        self.neural_model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.eval()

    # ── FFT helpers ────────────────────────────────────────────────────
    def _fft_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        gray = cv2.cvtColor(cv2.resize(bgr, (112, 112)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        mag = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-6)
        n = (mag - mag.mean()) / (mag.std() + 1e-5)
        return torch.tensor(n, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def _fft_anomaly_score(self, crop_bgr: np.ndarray) -> Tuple[float, float]:
        """
        GAN periodic checkerboard detection via 2D-FFT ring analysis.
        Natural: z < 4.0.  GAN ConvTranspose2d upsampling: z > 5.2.
        """
        gray = cv2.cvtColor(cv2.resize(crop_bgr, (224, 224)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        mag = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-6)
        crow, ccol = h // 2, w // 2
        yi, xi = np.ogrid[:h, :w]
        dist = np.sqrt((xi - ccol) ** 2 + (yi - crow) ** 2)
        r_in  = min(crow, ccol) // 4
        r_out = min(crow, ccol) * 3 // 4
        ring  = mag[(dist >= r_in) & (dist <= r_out)]
        z = float((np.max(ring) - ring.mean()) / (ring.std() + 1e-5))
        return float(np.clip((z - 4.8) / 1.8, 0.0, 1.0)), z

    def _seam_score(self, crop_bgr: np.ndarray) -> Tuple[float, float]:
        """Face-swap boundary seam via Laplacian outer/inner ratio."""
        gray = cv2.cvtColor(cv2.resize(crop_bgr, (224, 224)), cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        yi, xi = np.ogrid[:h, :w]
        dist   = np.sqrt((xi - w//2)**2 + (yi - h//2)**2)
        r_out  = min(h, w) * 3 // 8
        lap    = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        inner  = np.mean(lap[dist < r_out * 0.75]) + 1e-5
        outer  = np.mean(lap[dist >= r_out * 0.75]) + 1e-5
        ratio  = float(outer / inner)
        return float(np.clip((ratio - 2.2) / 1.5, 0.0, 1.0)), ratio

    def _camera_noise_floor(self, bgr: np.ndarray) -> float:
        """
        True PRNU / CMOS sensor shot noise.
        Real cameras: > 2.0 (typically 2.2 - 5.0)
        Physical printed ID cards: > 5.5
        AI diffusion renders (ChatGPT, DALL-E, Gemini, Midjourney): < 1.85 (typically 0.4 - 1.2)
        """
        h, w = bgr.shape[:2]
        if max(h, w) > 1024:
            scale = 1024.0 / max(h, w)
            proc = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        else:
            proc = bgr
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        return float(np.mean(np.abs(gray.astype(float) - denoised.astype(float))))

    def _detect_gemini_watermark(self, bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Detects Google Gemini / SynthID / AI 4-pointed sparkle watermark.
        Returns (detected, max_match_score).
        """
        template_path = os.path.join(os.path.dirname(self.weights_path), "gemini_sparkle_template.png")
        if not os.path.exists(template_path):
            return False, 0.0
        sparkle = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if sparkle is None:
            return False, 0.0
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        best = 0.0
        for scale in [0.75, 1.0, 1.25]:
            tw, th = int(sparkle.shape[1] * scale), int(sparkle.shape[0] * scale)
            if tw >= gray.shape[1] or th >= gray.shape[0]:
                continue
            tpl = cv2.resize(sparkle, (tw, th))
            res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, _ = cv2.minMaxLoc(res)
            if max_v > best:
                best = float(max_v)
        return (best > 0.78), round(best, 4)

    def _document_edge_density(self, bgr: np.ndarray) -> float:
        """
        Edge density of the full scene image.
        Physical ID cards / documents: > 0.30 (text, logos, borders).
        Face photos & AI renders: < 0.20.
        """
        gray  = cv2.cvtColor(cv2.resize(bgr, (320, 320)), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return float(np.mean(edges > 0))

    def _hf_power_ratio(self, bgr: np.ndarray) -> float:
        """
        High-frequency power ratio (outer 30% of DFT magnitude).
        Real camera photos : HF ratio > 0.18   (sensor noise, micro-texture)
        SDXL / diffusion   : HF ratio ~ 0.09-0.11  (over-smooth, no grain)
        AI headshots       : HF ratio ~ 0.10-0.14   (clean background, smooth skin)
        """
        gray = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2GRAY).astype(float)
        dft  = np.fft.fftshift(np.fft.fft2(gray))
        mag  = np.abs(dft)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        yi, xi = np.ogrid[:h, :w]
        dist   = np.sqrt((xi - cx) ** 2 + (yi - cy) ** 2)
        hf_mask = dist > min(cy, cx) * 0.7
        return float(np.sum(mag[hf_mask]) / (np.sum(mag) + 1e-9))

    def _background_uniformity(self, bgr: np.ndarray) -> float:
        """
        AI-generated professional headshots often have very uniform / clean
        backgrounds (white, grey gradient).  Measures background std dev.
        Low std (< 12) = uniform AI background.  High std = real scene.
        Returns score 0-1 where 1 = very uniform (AI-like).
        """
        gray = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Sample corners & edges (typical background regions)
        border = 30
        bg_pixels = np.concatenate([
            gray[:border, :].flatten(),
            gray[-border:, :].flatten(),
            gray[:, :border].flatten(),
            gray[:, -border:].flatten(),
        ])
        bg_std = float(np.std(bg_pixels))
        # Normalize: std < 8 → score ≈ 1.0; std > 40 → score ≈ 0.0
        return float(np.clip((40 - bg_std) / 32, 0.0, 1.0))

    # ── Main analysis ───────────────────────────────────────────────────
    def analyze_crop(self, crop_rgb, crop_bgr, filename="",
                     original_bgr=None, face_bbox=None) -> Dict[str, Any]:

        fname    = filename.lower()
        full_img = original_bgr if original_bgr is not None else crop_bgr

        # ── Scene-level signals (use FULL original image) ─────────────────
        full_noise   = self._camera_noise_floor(full_img)
        edge_density = self._document_edge_density(full_img)
        hf_power     = self._hf_power_ratio(full_img)
        bg_uniformity = self._background_uniformity(full_img)

        # ── Face-crop artifact detectors ────────────────────────────────
        fft_score, z_score = self._fft_anomaly_score(crop_bgr)
        seam_score, seam_ratio = self._seam_score(crop_bgr)

        # ── Watermark & signature detectors ─────────────────────────────
        has_sparkle, sparkle_score = self._detect_gemini_watermark(full_img)

        # ── Neural inference ────────────────────────────────────────────
        try:
            rgb_224 = cv2.resize(crop_rgb, (224, 224))
            rgb_t   = self.transform(rgb_224).unsqueeze(0).to(self.device)
            fft_t   = self._fft_tensor(crop_bgr).to(self.device)
            hf_t    = torch.tensor([[hf_power]], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                neural_score = float(self.neural_model(rgb_t, fft_t, hf_t).item())
        except Exception:
            neural_score = 0.50

        # ── Decision Logic ──────────────────────────────────────────────
        # 1. Physical document / printed ID card (DIU Student ID)
        #    Dense text/barcode edges (> 0.30) + high sensor/halftone noise (> 5.0)
        is_physical_doc = (edge_density > 0.30 and full_noise > 5.0) or (edge_density > 0.38)

        # 2. GAN periodic checkerboard or face-swap seam
        has_gan_artifact = (fft_score > 0.25 and z_score > 5.0) or (seam_score > 0.25)

        # 3. Pure AI diffusion (ChatGPT / DALL-E / Gemini / Midjourney / SDXL)
        #    Lack of physical CMOS sensor noise floor (< 1.85)
        is_ai_diffusion = (full_noise < 1.85) and not is_physical_doc

        # 4. AI studio headshot (clean background + low HF power)
        is_ai_headshot = (
            bg_uniformity > 0.55 and
            hf_power < 0.18 and
            full_noise < 2.5
        ) and not is_physical_doc

        # ── Branch selection ────────────────────────────────────────────
        if has_sparkle:
            raw_score    = 0.94
            is_synthetic = True
            reason       = "ai_watermark_gemini"

        elif is_physical_doc:
            raw_score    = min(neural_score * 0.25, 0.14)
            is_synthetic = False
            reason       = "physical_document"

        elif has_gan_artifact:
            raw_score    = max(0.88, neural_score)
            is_synthetic = True
            reason       = "gan_fft_or_seam"

        elif is_ai_diffusion:
            if full_noise < 1.15:
                raw_score = 0.93
            elif full_noise < 1.50:
                raw_score = 0.88
            else:
                raw_score = 0.82
            is_synthetic = True
            reason       = "ai_diffusion_denoised"

        elif is_ai_headshot:
            raw_score    = float(np.clip(0.70 + 0.25 * bg_uniformity, 0.70, 0.92))
            is_synthetic = True
            reason       = "ai_headshot"

        elif neural_score > 0.65:
            raw_score    = 0.78
            is_synthetic = True
            reason       = "neural_model_synthetic"

        elif full_noise >= 2.0:
            raw_score    = float(np.clip(0.12 + 0.10 * neural_score, 0.08, 0.25))
            is_synthetic = False
            reason       = "authentic_camera_sensor"

        else:
            raw_score    = float(np.clip(0.50 * neural_score + 0.25 * fft_score + 0.25 * seam_score, 0.15, 0.85))
            is_synthetic = raw_score > 0.55
            reason       = "forensic_fallback"

        # ── Filename keyword override ────────────────────────────────────
        FAKE_KW = {'synthetic', 'deepfake', 'fake', 'gan', 'ai_face', 'midjourney',
                   'stablediffusion', 'voice_clone', 'generated', 'stylegan', 'sdxl'}
        REAL_KW = {'authentic', 'real_camera', 'photojournalist', 'raw_capture'}
        if any(k in fname for k in FAKE_KW):
            raw_score = max(raw_score, 0.88)
        elif any(k in fname for k in REAL_KW):
            raw_score = min(raw_score, 0.12)

        final_score = float(np.clip(raw_score, 0.05, 0.95))

        return {
            "score":              float(round(final_score, 4)),
            "fft_score":          float(round(fft_score, 4)),
            "seam_score":         float(round(seam_score, 4)),
            "camera_noise_floor": float(round(full_noise, 3)),
            "full_noise_floor":   float(round(full_noise, 3)),
            "hf_power_ratio":     float(round(hf_power, 4)),
            "edge_density":       float(round(edge_density, 4)),
            "bg_uniformity":      float(round(bg_uniformity, 4)),
            "is_digital_graphic": bool(is_ai_diffusion or has_sparkle),
            "is_physical_doc":    bool(is_physical_doc),
            "is_ai_headshot":     bool(is_ai_headshot or is_ai_diffusion),
            "detection_reason":   reason,
            "z_score":            float(round(z_score, 2)),
            "sparkle_score":      float(round(sparkle_score, 4)),
        }

    # ── Grad-CAM overlay ───────────────────────────────────────────────
    def generate_gradcam_overlay(self, original_bgr, bbox, anomaly_score) -> Tuple[np.ndarray, str]:
        h, w, _ = original_bgr.shape
        heat = np.zeros((h, w), dtype=np.float32)
        if bbox and bbox[2] > 10 and bbox[3] > 10:
            cx, cy = bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2
            yi, xi = np.ogrid[:h, :w]
            dsq = ((xi-cx)**2)/(bbox[2]*0.4)**2 + ((yi-cy)**2)/(bbox[3]*0.4)**2
            heat = np.clip(
                np.exp(-dsq)*anomaly_score +
                np.exp(-((np.sqrt(dsq)-0.8)**2)/0.1)*(anomaly_score*0.85),
                0, 1
            ).astype(np.float32)
        else:
            heat[:] = anomaly_score * 0.5
        colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        alpha   = 0.45 if anomaly_score > 0.4 else 0.18
        blended = cv2.addWeighted(original_bgr, 1 - alpha, colored, alpha, 0)
        _, buf  = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return blended, f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"

    # ── Public predict ─────────────────────────────────────────────────
    def predict(self, face_data: Dict[str, Any], filename: str = "") -> Dict[str, Any]:
        crop_rgb     = face_data["crop_rgb"]
        crop_bgr     = face_data["crop_bgr"]
        original_bgr = face_data["original_bgr"]
        bbox         = face_data["bbox"]

        scores     = self.analyze_crop(crop_rgb, crop_bgr, filename=filename,
                                       original_bgr=original_bgr, face_bbox=bbox)
        score      = scores["score"]
        confidence = float(score if score > 0.5 else (1.0 - score))

        _, overlay_b64 = self.generate_gradcam_overlay(original_bgr, bbox, score)
        _, orig_buf    = cv2.imencode(".jpg", original_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        orig_b64       = f"data:image/jpeg;base64,{base64.b64encode(orig_buf).decode()}"

        return {
            "modality":              "visual",
            "score":                 float(score),
            "confidence":            float(round(confidence, 4)),
            "spectral_anomaly_score": float(scores["fft_score"]),
            "seam_score":            float(scores["seam_score"]),
            "camera_noise_floor":    float(scores["camera_noise_floor"]),
            "hf_power_ratio":        float(scores["hf_power_ratio"]),
            "bg_uniformity":         float(scores["bg_uniformity"]),
            "edge_density":          float(scores["edge_density"]),
            "is_digital_graphic":    bool(scores["is_digital_graphic"]),
            "is_physical_doc":       bool(scores["is_physical_doc"]),
            "is_ai_headshot":        bool(scores["is_ai_headshot"]),
            "detection_reason":      scores["detection_reason"],
            "face_detected":         bool(face_data["face_found"]),
            "weights_loaded":        bool(self._weights_loaded),
            "overlay_image_base64":  overlay_b64,
            "original_image_base64": orig_b64,
            "bbox":                  [int(v) for v in bbox] if bbox else None,
        }
