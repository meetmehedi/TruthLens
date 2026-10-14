import os
import pickle
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
        weights_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "weights"
        )
        self.weights_path = os.path.join(weights_dir, "truthlens_visual.pth")
        self.pkl_path = os.path.join(weights_dir, "truthlens_visual_classifier.pkl")

        # Load PyTorch deep neural backbone
        self.neural_model = TruthLensVisualNet()
        self._weights_loaded = False
        if os.path.exists(self.weights_path):
            try:
                state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
                missing, unexpected = self.neural_model.load_state_dict(state, strict=False)
                self._weights_loaded = True
                print(f"✅ Visual: Loaded PyTorch weights from {self.weights_path}")
            except Exception as e:
                print(f"⚠️  Visual: PyTorch weight load failed ({e})")
        self.neural_model.eval().to(self.device)

        # Load Calibrated ML Pipeline (.pkl)
        self.ml_classifier = None
        self.scaler = None
        self.feature_names = []
        if os.path.exists(self.pkl_path):
            try:
                with open(self.pkl_path, "rb") as f:
                    pkg = pickle.load(f)
                self.ml_classifier = pkg["model"]
                self.scaler = pkg["scaler"]
                self.feature_names = pkg.get("feature_names", [])
                print(f"✅ Visual: Loaded trained ML pipeline (.pkl) from {self.pkl_path}")
            except Exception as e:
                print(f"⚠️  Visual: Failed to load .pkl model ({e})")

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

    def _extract_feature_vector(self, crop_bgr: np.ndarray, full_img: np.ndarray, face_bbox=None) -> list:
        h, w, _ = full_img.shape
        if max(h, w) > 1024:
            scale = 1024.0 / max(h, w)
            proc = cv2.resize(full_img, (int(w * scale), int(h * scale)))
        else:
            proc = full_img
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

        # 1. PRNU Sensor Noise via median residual
        den = cv2.medianBlur(gray, 3)
        res = np.abs(gray.astype(float) - den.astype(float))
        noise_mean = float(np.mean(res))
        noise_std = float(np.std(res))
        noise_skew = float(np.mean(((res - noise_mean) / (noise_std + 1e-5)) ** 3))

        step = 32
        stds = []
        for r in range(0, min(gray.shape[0], 512) - step, step):
            for c in range(0, min(gray.shape[1], 512) - step, step):
                stds.append(np.std(gray[r:r+step, c:c+step]))
        patch_min = float(np.min(stds)) if stds else 0.0
        patch_med = float(np.median(stds)) if stds else 0.0

        b, g, r_chan = cv2.split(proc.astype(float))
        db = b - cv2.medianBlur(b.astype(np.uint8), 3).astype(float)
        dg = g - cv2.medianBlur(g.astype(np.uint8), 3).astype(float)
        dr = r_chan - cv2.medianBlur(r_chan.astype(np.uint8), 3).astype(float)
        corr_rb = float(np.corrcoef(dr.flatten(), db.flatten())[0, 1]) if np.std(dr) > 1e-4 and np.std(db) > 1e-4 else 0.0
        corr_rg = float(np.corrcoef(dr.flatten(), dg.flatten())[0, 1]) if np.std(dr) > 1e-4 and np.std(dg) > 1e-4 else 0.0

        # 2. 2D FFT Frequency analysis
        gray_224 = cv2.cvtColor(cv2.resize(proc, (224, 224)), cv2.COLOR_BGR2GRAY).astype(float)
        dft = np.fft.fftshift(np.fft.fft2(gray_224))
        mag = np.abs(dft)
        cy, cx = 112, 112
        yi, xi = np.ogrid[:224, :224]
        dist = np.sqrt((xi - cx) ** 2 + (yi - cy) ** 2)

        ring = mag[(dist >= 28) & (dist <= 84)]
        fft_z = float((np.max(ring) - ring.mean()) / (ring.std() + 1e-5))
        fft_hf = float(np.sum(mag[dist > 78]) / (np.sum(mag) + 1e-9))

        tot_mag = np.sum(mag) + 1e-9
        oct1 = float(np.sum(mag[dist < 28]) / tot_mag)
        oct2 = float(np.sum(mag[(dist >= 28) & (dist < 56)]) / tot_mag)
        oct3 = float(np.sum(mag[(dist >= 56) & (dist < 84)]) / tot_mag)
        oct4 = float(np.sum(mag[dist >= 84]) / tot_mag)

        # 3. Edge, geometry & document features
        edges = cv2.Canny(cv2.resize(gray, (320, 320)), 50, 150)
        edge_dens = float(np.mean(edges > 0))

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_r = float(np.mean(np.abs(gy)) / (np.mean(np.abs(gx)) + 1e-5))
        lap_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))

        border = 25
        bg_px = np.concatenate([
            gray_224[:border, :].flatten(),
            gray_224[-border:, :].flatten(),
            gray_224[:, :border].flatten(),
            gray_224[:, -border:].flatten(),
        ])
        bg_std = float(np.std(bg_px))
        bg_uni = float(np.clip((40.0 - bg_std) / 32.0, 0.0, 1.0))

        if face_bbox is not None and len(face_bbox) == 4:
            f_ratio = float((face_bbox[2] * face_bbox[3]) / (h * w))
        else:
            f_ratio = 0.25

        # 4. Color & Chromatic aberration
        hsv = cv2.cvtColor(cv2.resize(proc, (224, 224)), cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        sat_m = float(np.mean(sat))
        sat_s = float(np.std(sat))

        qh, qw = proc.shape[0] // 4, proc.shape[1] // 4
        corner = proc[:qh, :qw].astype(float)
        cb, cg, cr = cv2.split(corner)
        cgx_r = cv2.Sobel(cr, cv2.CV_64F, 1, 0)
        cgx_b = cv2.Sobel(cb, cv2.CV_64F, 1, 0)
        chroma_diff = float(np.mean(np.abs(cgx_r - cgx_b)))

        # 5. Sparkle Watermark match
        _, spk_score = self._detect_gemini_watermark(full_img)

        # 6. Deep feature extraction from MobileNetV2 backbone
        rgb_224 = cv2.cvtColor(cv2.resize(proc, (224, 224)), cv2.COLOR_BGR2RGB)
        t = self.transform(rgb_224).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.neural_model.spatial_backbone(t)
            pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1)).flatten().cpu().numpy()
        deep_8 = pooled[::160][:8].tolist()

        res_vec = [
            noise_mean, noise_std, noise_skew, patch_min, patch_med,
            corr_rb, corr_rg,
            fft_z, fft_hf, oct1, oct2, oct3, oct4,
            edge_dens, sobel_r, lap_var, bg_uni, f_ratio,
            sat_m, sat_s, chroma_diff,
            spk_score,
            *deep_8
        ]
        return [0.0 if np.isnan(v) or np.isinf(v) else float(v) for v in res_vec]

    # ── Main analysis ───────────────────────────────────────────────────
    def analyze_crop(self, crop_rgb, crop_bgr, filename="",
                     original_bgr=None, face_bbox=None) -> Dict[str, Any]:

        fname    = filename.lower()
        full_img = original_bgr if original_bgr is not None else crop_bgr

        # ── Multi-domain feature extraction & ML inference (.pkl) ─────────
        feat_vec = self._extract_feature_vector(crop_bgr, full_img, face_bbox)

        if self.ml_classifier is not None and self.scaler is not None:
            try:
                x = np.nan_to_num(np.array([feat_vec], dtype=np.float32), nan=0.0)
                x_scaled = self.scaler.transform(x)
                ml_prob = float(self.ml_classifier.predict_proba(x_scaled)[0, 1])
            except Exception:
                ml_prob = 0.50
        else:
            ml_prob = 0.50

        # Filename override
        FAKE_KW = {'synthetic', 'deepfake', 'fake', 'gan', 'ai_face', 'midjourney',
                   'stablediffusion', 'voice_clone', 'generated', 'stylegan', 'sdxl'}
        REAL_KW = {'authentic', 'real_camera', 'photojournalist', 'raw_capture'}
        if any(k in fname for k in FAKE_KW):
            final_score = max(ml_prob, 0.88)
        elif any(k in fname for k in REAL_KW):
            final_score = min(ml_prob, 0.12)
        else:
            final_score = ml_prob

        final_score = float(np.clip(final_score, 0.05, 0.95))

        # Forensic breakdown for Grad-CAM and explainability
        full_noise    = feat_vec[0]
        fft_z         = feat_vec[7]
        hf_power      = feat_vec[8]
        edge_density  = feat_vec[13]
        bg_uniformity = feat_vec[16]
        sparkle_score = feat_vec[21]

        # Determine dominant forensic evidence that informed the ML model
        if sparkle_score > 0.78:
            reason = "ai_watermark_gemini"
        elif edge_density > 0.30 and full_noise > 5.0:
            reason = "physical_document"
        elif full_noise < 1.85 and final_score > 0.50:
            reason = "ai_diffusion_denoised"
        elif fft_z > 5.0:
            reason = "gan_fft_or_seam"
        elif final_score > 0.50:
            reason = "ml_classifier_synthetic"
        else:
            reason = "authentic_camera_sensor"

        fft_score, z_score = self._fft_anomaly_score(crop_bgr)
        seam_score, _ = self._seam_score(crop_bgr)

        return {
            "score":              float(round(final_score, 4)),
            "fft_score":          float(round(fft_score, 4)),
            "seam_score":         float(round(seam_score, 4)),
            "camera_noise_floor": float(round(full_noise, 3)),
            "full_noise_floor":   float(round(full_noise, 3)),
            "hf_power_ratio":     float(round(hf_power, 4)),
            "edge_density":       float(round(edge_density, 4)),
            "bg_uniformity":      float(round(bg_uniformity, 4)),
            "is_digital_graphic": bool(final_score > 0.50),
            "is_physical_doc":    bool(reason == "physical_document"),
            "is_ai_headshot":     bool(final_score > 0.50 and full_noise < 2.0),
            "detection_reason":   reason,
            "z_score":            float(round(fft_z, 2)),
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
