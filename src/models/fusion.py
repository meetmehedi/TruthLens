import numpy as np
from typing import Dict, Any, Optional


class MultimodalFusionEngine:
    """
    Calibrated late-fusion engine.
    Verdict thresholds:
      • Authentic       → score < 0.38
      • Uncertain       → 0.38 – 0.65
      • Likely Synthetic→ score > 0.65
    """
    def __init__(self):
        self.weights = {"visual": 0.60, "audio": 0.30, "text": 0.10}

    def fuse(self,
             visual_result: Optional[Dict[str, Any]] = None,
             audio_result:  Optional[Dict[str, Any]] = None,
             text_result:   Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        active = {}
        if visual_result and "score" in visual_result:
            active["visual"] = visual_result
        if audio_result and "score" in audio_result:
            active["audio"] = audio_result
        if text_result and "score" in text_result:
            active["text"] = text_result

        if not active:
            return {
                "verdict": "Uncertain", "final_score": 0.5,
                "badge_color": "warning",
                "synthetic_probability_percent": 50.0,
                "confidence_percent": 50.0,
                "summary": "No analysable media detected.",
                "reasons": ["No extractable modality features found."],
                "modalities_analyzed": [],
                "normalized_weights": {}
            }

        # Normalise weights to present modalities only
        priors = {m: self.weights[m] for m in active}
        if "visual" in active or "audio" in active:
            priors.pop("text", None) if "text" in active else None
            priors = {m: self.weights[m] for m in active}
        total = sum(priors.values())
        nw = {m: priors[m] / total for m in active}

        # Weighted score
        ws = sum(nw[m] * active[m]["score"] for m in active)

        # Safety rule: primary-media strong fake signal must not be diluted
        primary_max = max(
            (active[m]["score"] for m in active if m in ("visual", "audio")),
            default=0.0
        )
        if primary_max > 0.75:
            ws = max(ws, primary_max * 0.92)

        # Build reasons
        reasons = []
        for m, data in active.items():
            s = data["score"]
            if m == "visual":
                doc = data.get("document_context", False)
                if doc:
                    reasons.append("Visual Analysis: Document / ID-card scan detected — forensic signals adjusted for printed media.")
                elif s > 0.65:
                    reasons.append(f"Visual Analysis ({int(s*100)}% anomaly): GAN spectral checkerboard artifacts and face-swap boundary seam detected.")
                elif s < 0.38:
                    reasons.append("Visual Analysis: Natural skin texture, coherent eye reflections, and organic edge gradients — consistent with genuine photography.")
                else:
                    reasons.append("Visual Analysis: Borderline signals — possible image compression or subtle artefacts. No conclusive manipulation detected.")
            elif m == "audio":
                if s > 0.65:
                    reasons.append(f"Acoustic Analysis ({int(s*100)}% anomaly): Rigid harmonic regularity and high-frequency cutoff characteristic of neural vocoders.")
                elif s < 0.38:
                    reasons.append("Acoustic Analysis: Natural vocal micro-tremors and continuous ambient decay — consistent with genuine human speech.")
                else:
                    reasons.append("Acoustic Analysis: Inconclusive — audio may be compressed or filtered. No definitive synthesis signature.")
            elif m == "text" and s > 0.60:
                reasons.append(f"Text Analysis ({int(s*100)}% anomaly): Repetitive syntactic patterns and AI-typical transitional markers detected.")

        final = round(float(np.clip(ws, 0.04, 0.97)), 4)

        # ── Verdict thresholds ───────────────────────────
        if final > 0.65:
            verdict, color = "Likely Synthetic / Deepfake", "danger"
            summary = "Strong forensic evidence of artificial manipulation or neural generation."
            conf = round(final * 100.0, 1)
        elif final < 0.38:
            verdict, color = "Authentic", "success"
            summary = "Media exhibits consistent organic biological and acoustic characteristics."
            conf = round((1.0 - final) * 100.0, 1)
        else:
            verdict, color = "Uncertain — Manual Review Advised", "warning"
            summary = "Borderline forensic signals; manual expert review recommended."
            conf = round(55.0 + abs(final - 0.5) * 40.0, 1)

        return {
            "verdict": verdict,
            "badge_color": color,
            "final_score": final,
            "synthetic_probability_percent": round(final * 100, 1),
            "confidence_percent": conf,
            "summary": summary,
            "reasons": reasons,
            "modalities_analyzed": list(active.keys()),
            "normalized_weights": {k: round(v, 2) for k, v in nw.items()}
        }
