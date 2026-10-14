import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_score, recall_score

class Evaluator:
    """
    Standard benchmark evaluation engine calculating ROC-AUC, F1, Accuracy,
    and Equal Error Rate (EER) as specified for FaceForensics++ and ASVspoof benchmarks.
    """
    @staticmethod
    def compute_eer(y_true: List[int], y_scores: List[float]) -> Tuple[float, float]:
        """
        Calculates Equal Error Rate (EER) where False Acceptance Rate (FAR) == False Rejection Rate (FRR).
        Standard benchmark metric for ASVspoof and biometric anti-spoofing.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1 - tpr
        
        # Find index where |fpr - fnr| is minimized
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        threshold = float(thresholds[idx])
        return round(eer, 4), round(threshold, 4)

    @staticmethod
    def compute_all_metrics(y_true: List[int], y_scores: List[float], threshold: float = 0.5) -> Dict[str, Any]:
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        eer, eer_thresh = Evaluator.compute_eer(y_true, y_scores)

        return {
            "accuracy": round(float(acc), 4),
            "f1_score": round(float(f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "roc_auc": round(float(roc_auc), 4),
            "equal_error_rate_eer": eer,
            "optimal_threshold": eer_thresh
        }
