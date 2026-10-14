import os
import glob
import requests

SERVER_URL = "http://localhost:8000/api/analyze"

test_cases = [
    # Visual Modality
    {
        "category": "Visual",
        "name": "Real Camera Selfie (User Photo)",
        "file": "/Users/md.mehedihasan/.gemini/antigravity-ide/brain/839bfd8a-bd01-4a5e-b3fb-34facbcb60d4/.user_uploaded/media_1791953311600.png",
        "expected": "Authentic"
    },
    {
        "category": "Visual",
        "name": "Physical Student ID Card (DIU Card)",
        "file": "extracted_id_card.jpg",
        "expected": "Authentic"
    },
    {
        "category": "Visual",
        "name": "FFHQ Real Face (Benchmark #0010)",
        "file": "data/real_faces/real_face_0010.jpg",
        "expected": "Authentic"
    },
    {
        "category": "Visual",
        "name": "FFHQ Real Face (Benchmark #0240)",
        "file": "data/real_faces/real_face_0240.jpg",
        "expected": "Authentic"
    },
    {
        "category": "Visual",
        "name": "StyleGAN Fake Face (Benchmark #0010)",
        "file": "data/fake_faces/fake_face_0010.jpg",
        "expected": "Likely Synthetic / Deepfake"
    },
    {
        "category": "Visual",
        "name": "StyleGAN Fake Face (Benchmark #0240)",
        "file": "data/fake_faces/fake_face_0240.jpg",
        "expected": "Likely Synthetic / Deepfake"
    },
    # Audio Modality
    {
        "category": "Audio",
        "name": "Human Natural Speech (16kHz WAV)",
        "file": "data/real_audio/real_audio_0000.wav",
        "expected": "Authentic"
    },
    {
        "category": "Audio",
        "name": "AI Voice Clone (Vocoder Synth WAV)",
        "file": "data/fake_audio/fake_audio_0000.wav",
        "expected": "Likely Synthetic / Deepfake"
    },
    # Text Modality
    {
        "category": "Text",
        "name": "Human Team Meeting Message",
        "text": "Hey everyone, we had a team meeting today at 4pm and went over the presentation slides.",
        "expected": "Authentic"
    },
    {
        "category": "Text",
        "name": "AI Generated Cliché Text",
        "text": "In conclusion, it is important to remember that this is a testament to the tapestry of innovation. We must delve into this beacon of hope.",
        "expected": "Likely Synthetic / Deepfake"
    }
]

print("=" * 105)
print(f"{'CATEGORY':<9} | {'TEST CASE NAME':<38} | {'VERDICT':<27} | {'SYNTH %':<8} | {'STATUS'}")
print("=" * 105)

passed = 0
total = len(test_cases)

for t in test_cases:
    cat = t["category"]
    name = t["name"]
    exp = t["expected"]
    
    if "file" in t:
        if not os.path.exists(t["file"]):
            print(f"{cat:<9} | {name:<38} | FILE NOT FOUND ({t['file'][:25]}...)")
            continue
        with open(t["file"], "rb") as f:
            res = requests.post(SERVER_URL, files={"file": f})
    else:
        res = requests.post(SERVER_URL, data={"text_content": t["text"]})
        
    if res.status_code == 200:
        d = res.json()
        verdict = d.get("verdict", "Unknown")
        synth = d.get("synthetic_probability_percent", 0.0)
        match = (exp in verdict) or (verdict in exp)
        status = "✅ PASS" if match else "❌ FAIL"
        if match:
            passed += 1
        print(f"{cat:<9} | {name:<38} | {verdict:<27} | {synth:5.1f}%  | {status}")
    else:
        print(f"{cat:<9} | {name:<38} | ERROR {res.status_code}: {res.text[:30]}")

print("=" * 105)
print(f"Summary: {passed}/{total} Test Cases Passed ({passed/total*100:.1f}% Success Rate)")
print("=" * 105)
