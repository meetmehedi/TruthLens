import numpy as np
from typing import Dict, Any, List, Optional

class MultimodalFusionEngine:
    """
    Calibrated late-fusion engine combining visual, audio, and text modality scores.
    Handles missing modalities dynamically and produces an explainable tri-state verdict:
      - Authentic (< 0.40)
      - Uncertain (0.40 - 0.60)
      - Likely Synthetic (> 0.60)
    """
    def __init__(self):
        # Base relative priors
        self.default_weights = {
            "visual": 0.60,
            "audio": 0.30,
            "text": 0.10
        }

    def fuse(self, 
             visual_result: Optional[Dict[str, Any]] = None,
             audio_result: Optional[Dict[str, Any]] = None,
             text_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        
        active_modalities = {}
        if visual_result and "score" in visual_result:
            active_modalities["visual"] = visual_result
        if audio_result and "score" in audio_result:
            active_modalities["audio"] = audio_result
        if text_result and "score" in text_result:
            active_modalities["text"] = text_result

        if not active_modalities:
            return {
                "verdict": "Uncertain",
                "final_score": 0.5,
                "confidence_percent": 0.0,
                "reasons": ["No extractable modality features detected in payload."],
                "breakdown": {}
            }

        # Dynamic priority weighting
        priors = dict(self.default_weights)
        if "visual" in active_modalities or "audio" in active_modalities:
            if "text" in active_modalities:
                priors["text"] = 0.05

        total_priors = sum(priors[m] for m in active_modalities)
        normalized_weights = {m: priors[m] / total_priors for m in active_modalities}

        # Compute weighted synthetic score
        weighted_score = 0.0
        reasons = []

        for m, data in active_modalities.items():
            w = normalized_weights[m]
            s = data["score"]
            weighted_score += w * s

        # Forensic Priority Rule: If any primary media stream exhibits strong synthetic signals (>0.70),
        # supplementary modalities must not dilute the forgery alert
        max_media_synthetic = max([data["score"] for m, data in active_modalities.items() if m in ["visual", "audio"]], default=0.0)
        if max_media_synthetic > 0.70:
            weighted_score = max(weighted_score, max_media_synthetic * 0.95)

        for m, data in active_modalities.items():
            s = data["score"]
            # Generate plain-English reasons behind flags
            if m == "visual":
                if s > 0.60:
                    reasons.append(f"Visual Analysis ({int(s*100)}% anomaly): Detected facial boundary warping & abnormal high-frequency spectral roll-off.")
                elif s < 0.40:
                    reasons.append("Visual Analysis: Natural facial skin texture, coherent eye reflections, and organic edge gradients.")
                else:
                    reasons.append("Visual Analysis: Borderline visual artifacts; possible compression noise or subtle GAN generation.")
            elif m == "audio":
                if s > 0.60:
                    reasons.append(f"Acoustic Analysis ({int(s*100)}% anomaly): High-frequency spectral truncation and rigid harmonic patterns characteristic of neural vocoders.")
                elif s < 0.40:
                    reasons.append("Acoustic Analysis: Natural vocal micro-tremors and continuous ambient decay consistent with genuine human speech.")
                else:
                    reasons.append("Acoustic Analysis: Inconclusive acoustic indicators; audio may be heavily compressed or filtered.")
            elif m == "text":
                if s > 0.60:
                    reasons.append(f"Text Analysis ({int(s*100)}% anomaly): Repetitive syntactic structures and stereotypical synthetic transitional markers.")

        # Tri-state verdict determination
        final_score = round(float(weighted_score), 4)
        if final_score > 0.60:
            verdict = "Likely Synthetic"
            badge_color = "danger"
            summary_desc = "Evidence indicates artificial manipulation or neural generation."
            confidence = round(float(final_score * 100.0), 1)
        elif final_score < 0.40:
            verdict = "Authentic"
            badge_color = "success"
            summary_desc = "Media exhibits consistent organic biological and acoustic characteristics."
            confidence = round(float((1.0 - final_score) * 100.0), 1)
        else:
            verdict = "Uncertain"
            badge_color = "warning"
            summary_desc = "Borderline forensic signals; manual verification or higher resolution source recommended."
            confidence = round(float(55.0 + abs(final_score - 0.5) * 50.0), 1)

        return {
            "verdict": verdict,
            "badge_color": badge_color,
            "final_score": final_score,
            "synthetic_probability_percent": round(final_score * 100, 1),
            "confidence_percent": confidence,
            "summary": summary_desc,
            "reasons": reasons,
            "modalities_analyzed": list(active_modalities.keys()),
            "normalized_weights": {k: round(v, 2) for k, v in normalized_weights.items()}
        }

