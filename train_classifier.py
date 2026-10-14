import os
import glob
import pickle
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score

print("=" * 70)
print("TruthLens: Training Visual Forensic Classifier (.pkl pipeline)")
print("=" * 70)

# Pretrained MobileNetV2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features.to(device).eval()
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Sparkle template
template_path = "weights/gemini_sparkle_template.png"
if os.path.exists(template_path):
    sparkle_tpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
else:
    sparkle_tpl = None

FEATURE_NAMES = [
    "noise_mean", "noise_std", "noise_skew", "patch_min_std", "patch_median_std",
    "channel_corr_rb", "channel_corr_rg",
    "fft_z_score", "fft_hf_ratio", "fft_octave_1", "fft_octave_2", "fft_octave_3", "fft_octave_4",
    "edge_density", "sobel_ratio", "laplacian_var", "bg_uniformity", "face_ratio",
    "sat_mean", "sat_std", "chromatic_diff",
    "sparkle_score",
    "deep_f1", "deep_f2", "deep_f3", "deep_f4", "deep_f5", "deep_f6", "deep_f7", "deep_f8"
]

def extract_forensic_features(img_path, face_bbox=None):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w, _ = img.shape

    if max(h, w) > 1024:
        scale = 1024.0 / max(h, w)
        proc = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        proc = img
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

    # 1. PRNU Sensor Noise
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
    spk_score = 0.0
    if sparkle_tpl is not None:
        for sc in [0.75, 1.0, 1.25]:
            tw, th = int(sparkle_tpl.shape[1] * sc), int(sparkle_tpl.shape[0] * sc)
            if tw < gray.shape[1] and th < gray.shape[0]:
                tpl = cv2.resize(sparkle_tpl, (tw, th))
                rm = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_v, _, _ = cv2.minMaxLoc(rm)
                if max_v > spk_score:
                    spk_score = float(max_v)

    # 6. Deep feature extraction from MobileNetV2
    rgb_224 = cv2.cvtColor(cv2.resize(proc, (224, 224)), cv2.COLOR_BGR2RGB)
    t = img_transform(rgb_224).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = mobilenet(t)
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


if __name__ == "__main__":
    cache_path = "data/features_cache.npz"
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        data = np.load(cache_path)
        X, y = data["X"], data["y"]
    else:
        real_files = sorted(glob.glob("data/real_faces/*.*"))
        fake_files = sorted(glob.glob("data/fake_faces/*.*"))

        print(f"Found {len(real_files)} real samples and {len(fake_files)} fake samples.")
        X, y = [], []
        print("Extracting features from real images...")
        for p in real_files:
            feats = extract_forensic_features(p)
            if feats is not None:
                X.append(feats)
                y.append(0)

        print("Extracting features from synthetic/fake images...")
        for p in fake_files:
            feats = extract_forensic_features(p)
            if feats is not None:
                X.append(feats)
                y.append(1)

        X = np.nan_to_num(np.array(X, dtype=np.float32), nan=0.0)
        y = np.array(y, dtype=np.int32)
        np.savez(cache_path, X=X, y=y)
        print(f"Saved feature cache to {cache_path}")

    print(f"Dataset matrix: X={X.shape}, y={y.shape} (Authentic: {np.sum(y == 0)}, Synthetic: {np.sum(y == 1)})")

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Use pure Python gradient boosting + random forest (100% stable, no thread lock)
    gb = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )

    ensemble = VotingClassifier(
        estimators=[("gb", gb), ("rf", rf)],
        voting="soft"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring="roc_auc")
    print(f"5-Fold Cross-Validation ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Fit ensemble model directly on all data
    ensemble.fit(X_scaled, y)

    y_pred_proba = ensemble.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Authentic", "Synthetic"]))
    print(f"Full-set ROC-AUC: {roc_auc_score(y, y_pred_proba):.4f}")

    # Package and save to .pkl
    model_package = {
        "model": ensemble,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "version": "2.0.0",
        "architecture": "VotingClassifier(GradientBoostingClassifier + RandomForestClassifier)",
        "cv_roc_auc": float(np.mean(cv_scores))
    }

    os.makedirs("weights", exist_ok=True)
    pkl_path = "weights/truthlens_visual_classifier.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model_package, f)

    print(f"\n✅ Successfully saved trained ML model to: {pkl_path}")
    print(f"File size: {os.path.getsize(pkl_path) / 1024:.1f} KB")
