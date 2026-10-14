import re
import numpy as np
from typing import Dict, Any, List

class TextDetector:
    """
    Analyzes accompanying textual transcripts, captions, or statements
    for synthetic generative characteristics (perplexity variance, repetition,
    hallmark synthetic transitional phrases).
    """
    def __init__(self):
        self.ai_markers = [
            "delve into", "testament to", "it is important to remember",
            "in conclusion", "furthermore", "moreover", "beacon of hope",
            "tapestry of", "underscores the importance", "deep dive"
        ]

    def analyze_tokens(self, text: str) -> Dict[str, Any]:
        if not text or len(text.strip()) == 0:
            return {"score": 0.5, "confidence": 0.0, "flagged_tokens": [], "token_count": 0}

        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 5:
            return {"score": 0.5, "confidence": 0.1, "flagged_tokens": [], "token_count": len(words)}

        # 1. Lexical diversity (Type-Token Ratio)
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        # 2. Cliché AI marker phrase count
        marker_hits = []
        lowered = text.lower()
        for marker in self.ai_markers:
            if marker in lowered:
                marker_hits.append(marker)

        # 3. Sentence length burstiness (variance of sentence lengths)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences]
            burstiness = float(np.std(lengths))
        else:
            burstiness = 5.0

        # LLMs often have very uniform sentence lengths (low burstiness < 3.5)
        burstiness_score = 0.75 if burstiness < 3.0 else 0.3

        # Combine metrics
        marker_score = min(1.0, len(marker_hits) * 0.3)
        ttr_score = 0.7 if ttr < 0.6 else 0.3

        final_score = float(0.4 * marker_score + 0.3 * burstiness_score + 0.3 * ttr_score)
        confidence = float(np.abs(final_score - 0.5) * 2.0)

        return {
            "score": round(final_score, 4),
            "confidence": round(confidence, 4),
            "marker_hits": marker_hits,
            "burstiness": round(burstiness, 2),
            "type_token_ratio": round(ttr, 2),
            "token_count": len(words)
        }

    def predict(self, text: str) -> Dict[str, Any]:
        result = self.analyze_tokens(text)
        return {
            "modality": "text",
            "score": result["score"],
            "confidence": result["confidence"],
            "features": {
                "burstiness_index": result["burstiness"],
                "lexical_richness": result["type_token_ratio"],
                "synthetic_phrases": result["marker_hits"]
            }
        }
