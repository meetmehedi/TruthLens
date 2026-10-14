import cv2, os, glob
import numpy as np
from src.preprocessing.face_extractor import FaceExtractor

extractor = FaceExtractor()

def analyze_img(path):
    img = cv2.imread(path)
    if img is None: return None
    h, w, _ = img.shape
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(np.var(lap))
    
    denoised = cv2.medianBlur(gray, 3)
    noise_resid = np.abs(gray.astype(float) - denoised.astype(float))
    noise_mean = float(np.mean(noise_resid))
    noise_std = float(np.std(noise_resid))
    
    dft = np.fft.fftshift(np.fft.fft2(cv2.resize(gray, (224, 224)).astype(float)))
    dft_mag = np.abs(dft)
    cy, cx = 112, 112
    yi, xi = np.ogrid[:224, :224]
    dist = np.sqrt((xi - cx)**2 + (yi - cy)**2)
    hf_ratio = float(np.sum(dft_mag[dist > 78]) / (np.sum(dft_mag) + 1e-9))
    
    # Face crop characteristics
    fdata = extractor.extract_from_image(path)
    fc_noise = 0.0
    fc_lap = 0.0
    face_ratio = 0.0
    if fdata.get("face_found"):
        bbox = fdata.get("bbox", [0,0,0,0])
        face_ratio = (bbox[2] * bbox[3]) / (h * w) if bbox else 0.0
        fgray = cv2.cvtColor(cv2.resize(fdata["crop_bgr"], (224, 224)), cv2.COLOR_BGR2GRAY)
        fblur = cv2.GaussianBlur(fgray, (5, 5), 0)
        fc_noise = float(np.mean(np.abs(fgray.astype(float) - fblur.astype(float))))
        fc_lap = float(np.var(cv2.Laplacian(fgray, cv2.CV_64F)))
        
    return {
        "lap_v": lap_var,
        "noise_m": noise_mean,
        "noise_s": noise_std,
        "hf_ratio": hf_ratio,
        "fc_noise": fc_noise,
        "fc_lap": fc_lap,
        "face_ratio": face_ratio
    }

images = [
    ("REAL_ID_CARD", "/tmp/diu_id_card_test.jpg"),
    ("REAL_SELFIE", "/tmp/test_real_selfie.jpg"),
    ("AI_SUIT_GEMINI", "/Users/md.mehedihasan/Downloads/photo_2026-10-14_12-11-21.jpg"),
    ("AI_SUIT_GLASSES", "/Users/md.mehedihasan/Downloads/file_0000000092608211989c53feb22a3bc2.png"),
    ("AI_GROUP_HIJAB", "/Users/md.mehedihasan/Downloads/801892463_2944656869213752_8051524628313581501_n.jpg"),
    ("AI_GROUP_2", "/Users/md.mehedihasan/Downloads/801963996_1721063675617355_1344911804837458419_n.jpg"),
    ("AI_SCENE_1", "/Users/md.mehedihasan/Downloads/836635180_2881534232231389_6215037686487301998_n.jpg"),
    ("AI_SCENE_2", "/Users/md.mehedihasan/Downloads/839691663_2881534152231397_3499161189089261241_n.jpg"),
]

for label, p in images:
    if os.path.exists(p):
        r = analyze_img(p)
        print(f"{label:16s} | noise_m: {r['noise_m']:5.2f} | lap_v: {r['lap_v']:7.1f} | hf: {r['hf_ratio']:5.3f} | fc_noise: {r['fc_noise']:5.2f} | fc_lap: {r['fc_lap']:7.1f} | f_ratio: {r['face_ratio']:.3f}")
