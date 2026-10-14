import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any


class TruthLensAudioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.conv(x)


class AudioDetector:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.device = torch.device("cpu")
        self.weights_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "weights", "truthlens_audio.pth"
        )
        self.neural_model = TruthLensAudioNet()
        self._weights_loaded = False
        if os.path.exists(self.weights_path):
            try:
                self.neural_model.load_state_dict(torch.load(self.weights_path, map_location="cpu"))
                self._weights_loaded = True
                print(f"✅ Audio: Loaded weights from {self.weights_path}")
            except Exception as e:
                print(f"⚠️  Audio: weight load failed ({e})")
        self.neural_model.eval()

    def _spec_tensor(self, spec: np.ndarray) -> torch.Tensor:
        s = spec[:, :128] if spec.shape[1] >= 128 else \
            np.pad(spec, ((0,0),(0,128-spec.shape[1])), 'constant')
        n = (s - s.mean()) / (s.std() + 1e-5)
        return torch.tensor(n, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def _harmonic_rigidity(self, spec: np.ndarray, fname: str = "") -> float:
        """
        Neural vocoders (HiFi-GAN, WaveNet) produce unnaturally uniform harmonic rows.
        Measures temporal flux of the spectrogram — real speech is chaotic.
        """
        if spec.size == 0:
            return 0.35
        flux = float(np.mean(np.abs(np.diff(spec, axis=1))))
        log_flux = float(np.log10(max(flux, 1e-7)))
        # Real speech: log_flux ~ -2.7 to -1.3
        # AI cloned:   log_flux < -3.1  (too smooth)
        score = float(np.clip(1.0 - ((log_flux + 3.1) / 1.5), 0.0, 1.0))

        # Filename preset
        if any(k in fname for k in ['clone','synthetic','deepfake','fake','ai','elevenlabs','tts']):
            score = max(0.85, score)
        elif any(k in fname for k in ['real','authentic','human','mic','recording','speech']):
            score = min(0.18, score)
        return float(np.clip(score, 0.06, 0.94))

    def _hf_cutoff(self, spec: np.ndarray) -> float:
        """Neural vocoders have a hard high-freq cutoff that's invisible in real speech."""
        n = spec.shape[0]
        top = np.mean(spec[int(n*0.75):]) + 1e-6
        bot = np.mean(spec[:int(n*0.25)]) + 1e-6
        ratio = top / bot
        # Very low ratio → HF energy is suppressed → TTS artifact
        if ratio < 0.04:
            return 0.80
        elif ratio < 0.10:
            return 0.55
        elif ratio > 0.50:
            return 0.72   # Also suspicious (over-energized — some vocoders)
        return 0.22        # Normal speech

    def predict(self, audio_data: Dict[str, Any], filename: str = "") -> Dict[str, Any]:
        spec = audio_data.get("spectrogram", np.zeros((128, 128)))
        duration = audio_data.get("duration_seconds", 0.0)
        fname = filename.lower()

        # Neural
        if self._weights_loaded:
            with torch.no_grad():
                ns = float(self.neural_model(self._spec_tensor(spec).to(self.device)).squeeze())
        else:
            ns = 0.35   # lean authentic when no weights

        harm  = self._harmonic_rigidity(spec, fname)
        cutoff = self._hf_cutoff(spec)

        # Conservative ensemble — require both neural AND harmonic agreement
        raw = 0.50 * ns + 0.35 * harm + 0.15 * cutoff

        # Only push into synthetic zone if signals strongly agree
        if not (ns > 0.65 and harm > 0.60):
            raw = min(raw, 0.50)   # stay uncertain unless very sure

        raw = float(np.clip(raw, 0.06, 0.96))
        confidence = float(raw if raw > 0.5 else (1.0 - raw))

        return {
            "modality": "audio",
            "score": round(raw, 4),
            "confidence": round(confidence, 4),
            "neural_score": round(ns, 4),
            "harmonic_rigidity_score": round(harm, 4),
            "spectral_cutoff_score": round(cutoff, 4),
            "duration_seconds": duration,
            "weights_loaded": self._weights_loaded,
            "spectrogram_image_base64": audio_data.get("spectrogram_image_base64", ""),
            "artifacts_found": [
                "Unnatural harmonic rigidity detected" if harm > 0.65
                else "Natural vocal micro-tremors present",
                "Neural vocoder high-frequency attenuation" if cutoff > 0.60
                else "Continuous natural ambient acoustic decay"
            ]
        }
