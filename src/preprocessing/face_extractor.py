import cv2
import numpy as np
from PIL import Image
import os
import tempfile
from typing import List, Tuple, Dict, Any, Optional

class FaceExtractor:
    """
    Extracts face bounding boxes and crops from images and video streams.
    Uses OpenCV Haar Cascades with fallback to high-saliency central crop.
    """
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.face_cascade = None

    def detect_faces(self, bgr_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Returns list of (x, y, w, h) bounding boxes."""
        if self.face_cascade is None or bgr_image is None or bgr_image.size == 0:
            return []
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(25, 25)
        )
        return [tuple(f) for f in faces]

    def extract_from_image(self, image_input: Any) -> Dict[str, Any]:
        """
        Process an image (filepath, PIL Image, or numpy array).
        Returns cropped face, bounding box, and original BGR image.
        """
        if isinstance(image_input, str):
            bgr = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            bgr = image_input
        else:
            raise ValueError("Unsupported image input type")

        if bgr is None:
            raise ValueError("Failed to decode image input")

        h, w, _ = bgr.shape
        faces = self.detect_faces(bgr)

        if len(faces) > 0:
            # Pick largest detected face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, fw, fh = faces[0]
            # Add margin
            margin_x = int(0.15 * fw)
            margin_y = int(0.15 * fh)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w, x + fw + margin_x)
            y2 = min(h, y + fh + margin_y)
            crop = bgr[y1:y2, x1:x2]
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            face_found = True
        else:
            # Fallback to centered square crop
            min_dim = min(h, w)
            sy = (h - min_dim) // 2
            sx = (w - min_dim) // 2
            crop = bgr[sy:sy + min_dim, sx:sx + min_dim]
            bbox = (int(sx), int(sy), int(min_dim), int(min_dim))
            face_found = False

        resized_crop = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_AREA)
        rgb_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)

        return {
            "face_found": bool(face_found),
            "bbox": [int(x) for x in bbox],
            "crop_bgr": resized_crop,
            "crop_rgb": rgb_crop,
            "original_bgr": bgr,
            "dimensions": (int(w), int(h))
        }

    def extract_from_video(self, video_path: str, max_frames: int = 16) -> List[Dict[str, Any]]:
        """
        Samples frames uniformly across video duration and extracts face data.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = max_frames

        step = max(1, total_frames // max_frames)
        sampled_frames = []

        frame_idx = 0
        while cap.isOpened() and len(sampled_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                extracted = self.extract_from_image(frame)
                extracted["frame_index"] = frame_idx
                sampled_frames.append(extracted)
            frame_idx += 1

        cap.release()
        return sampled_frames
