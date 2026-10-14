import os, cv2
from src.models.visual import VisualDetector
from src.models.fusion import MultimodalFusionEngine
from src.preprocessing.face_extractor import FaceExtractor

visual = VisualDetector()
fusion = MultimodalFusionEngine()
extractor = FaceExtractor()

test_cases = [
    ("REAL_ID_CARD", "/tmp/diu_id_card_test.jpg"),
    ("REAL_SELFIE", "/tmp/test_real_selfie.jpg"),
    ("AI_SUIT_GEMINI", "/Users/md.mehedihasan/Downloads/photo_2026-10-14_12-11-21.jpg"),
    ("AI_SUIT_GLASSES", "/Users/md.mehedihasan/Downloads/file_0000000092608211989c53feb22a3bc2.png"),
    ("AI_GROUP_HIJAB", "/Users/md.mehedihasan/Downloads/801892463_2944656869213752_8051524628313581501_n.jpg"),
]

print("=" * 80)
print(f"{'LABEL':16s} | {'VERDICT':28s} | {'SYNTH%':7s} | {'REASON CODE':24s}")
print("=" * 80)

for label, p in test_cases:
    if not os.path.exists(p):
        print(f"Skipping {p} (not found)")
        continue
    fdata = extractor.extract_from_image(p)
    vres = visual.predict(fdata, filename=os.path.basename(p))
    fres = fusion.fuse(visual_result=vres)
    v = fres["verdict"]
    pct = fres["synthetic_probability_percent"]
    r_code = vres["detection_reason"]
    print(f"{label:16s} | {v:28s} | {pct:6.1f}% | {r_code:24s}")
    for r in fres["reasons"]:
        print(f"    -> {r}")
