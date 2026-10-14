import requests, os

url = "http://localhost:8000/api/analyze"

test_paths = [
    "/tmp/diu_id_card_test.jpg",
    "/tmp/test_real_selfie.jpg",
    "/Users/md.mehedihasan/Downloads/photo_2026-10-14_12-11-21.jpg",
    "/Users/md.mehedihasan/Downloads/file_0000000092608211989c53feb22a3bc2.png",
    "/Users/md.mehedihasan/Downloads/801892463_2944656869213752_8051524628313581501_n.jpg"
]

print("=" * 80)
print(f"{'FILENAME':35s} | {'VERDICT':28s} | {'SYNTH%':7s}")
print("=" * 80)

for p in test_paths:
    if not os.path.exists(p): continue
    name = os.path.basename(p)
    with open(p, "rb") as f:
        r = requests.post(url, files={"file": (name, f, "image/jpeg")})
        res = r.json()
        v = res.get("verdict", "N/A")
        s = res.get("synthetic_probability_percent", 0.0)
        reasons = res.get("reasons", [])
        print(f"{name:35s} | {v:28s} | {s:6.1f}%")
        for reason in reasons:
            print(f"    -> {reason}")
