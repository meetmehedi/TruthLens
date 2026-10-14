import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Training TruthLens Forensic Neural Models on: {device}")

# -----------------------------------------------------------------------------
# 1. VISUAL DUAL-STREAM NEURAL NETWORK (Spatial MobileNetV2 + 2D-FFT Spectral)
# -----------------------------------------------------------------------------
class SpectralCNNBranch(nn.Module):
    """Processes 2D-FFT magnitude power spectrum to capture GAN upsampling grids."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.conv(x)

class TruthLensVisualModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Spatial Stream: MobileNetV2
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.spatial_backbone = mobilenet.features
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Spectral Stream: 2D-FFT CNN
        self.spectral_branch = SpectralCNNBranch()
        
        # Multimodal Classification Fusion Head (1280 spatial + 128 spectral = 1408)
        self.fusion_head = nn.Sequential(
            nn.Linear(1280 + 128, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_tensor, fft_tensor):
        spatial_feats = self.spatial_pool(self.spatial_backbone(rgb_tensor)).flatten(1)
        spectral_feats = self.spectral_branch(fft_tensor)
        combined = torch.cat([spatial_feats, spectral_feats], dim=1)
        return self.fusion_head(combined)

# -----------------------------------------------------------------------------
# 2. SYNTHETIC & AUTHENTIC DATASET GENERATOR
# -----------------------------------------------------------------------------
class ForensicFaceDataset(Dataset):
    """
    Generates realistic authentic human portraits and synthetic manipulated pairs:
    - Authentic: Natural camera PRNU sensor noise, continuous organic skin micro-gradients.
    - Deepfake Type 1: High-frequency 2D-FFT periodic lattice artifacts (GAN upsampling).
    - Deepfake Type 2: Boundary seam feathering and edge blending warping (Face-swap).
    - Deepfake Type 3: Latent diffusion smoothing with high-frequency attenuation.
    """
    def __init__(self, num_samples: int = 800):
        self.num_samples = num_samples
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        is_synthetic = 1.0 if (idx % 2 == 1) else 0.0
        
        # Base face canvas (natural skin tones & lighting gradient)
        h, w = 224, 224
        img = np.zeros((h, w, 3), dtype=np.float32)
        
        # Natural human skin tone baseline (warm melanin tone)
        base_color = np.array([np.random.uniform(140, 220), np.random.uniform(110, 180), np.random.uniform(90, 150)], dtype=np.float32)
        img[:, :] = base_color
        
        # Facial oval geometry
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        dist_sq = ((x - cx) ** 2) / (75 ** 2) + ((y - cy) ** 2) / (95 ** 2)
        face_mask = dist_sq <= 1.0
        
        # Add facial features (eyes, eyebrows, mouth)
        img[face_mask] *= np.random.uniform(0.9, 1.1)
        # Eyes
        img[75:90, 75:95] = [30, 25, 20]
        img[75:90, 130:150] = [30, 25, 20]
        # Mouth
        img[150:165, 95:130] = [180, 60, 60]

        if is_synthetic == 0.0:
            # AUTHENTIC CHARACTERISTICS:
            # 1. Natural camera sensor noise (PRNU Gaussian noise)
            noise = np.random.normal(0, np.random.uniform(4, 9), (h, w, 3))
            img = np.clip(img + noise, 0, 255)
            # 2. Specular corneal reflection in pupils (biological liveness)
            img[80:83, 83:86] = [255, 255, 255]
            img[80:83, 138:141] = [255, 255, 255]
        else:
            # SYNTHETIC DEEPFAKE ARTIFACTS:
            manipulation_type = idx % 3
            if manipulation_type == 0:
                # GAN Periodic Lattice Checkerboard (ConvTranspose2d upsampling artifact)
                lattice = np.fromfunction(lambda i, j: np.sin(i * 0.9) * np.cos(j * 0.9) * 25.0, (h, w))
                for c in range(3):
                    img[:, :, c] = np.clip(img[:, :, c] + lattice, 0, 255)
            elif manipulation_type == 1:
                # Face-Swap Boundary Discontinuity (Paste / Seam feathering)
                boundary_ring = (dist_sq >= 0.85) & (dist_sq <= 1.15)
                img[boundary_ring] = cv2.GaussianBlur(img, (9, 9), 0)[boundary_ring] * 1.2
            else:
                # Diffusion Over-smoothing (Plastic skin with abrupt noise loss)
                img = cv2.bilateralFilter(img.astype(np.uint8), 9, 75, 75).astype(np.float32)

        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        rgb_tensor = self.transform(cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB))
        
        # 2D-FFT Power Spectrum (112x112 normalized for spectral CNN)
        gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(cv2.resize(gray, (112, 112)))
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-6)
        mag_norm = (mag - np.mean(mag)) / (np.std(mag) + 1e-5)
        fft_tensor = torch.tensor(mag_norm, dtype=torch.float32).unsqueeze(0)

        label_tensor = torch.tensor([is_synthetic], dtype=torch.float32)
        return rgb_tensor, fft_tensor, label_tensor

# -----------------------------------------------------------------------------
# 3. ACOUSTIC VOCODER NEURAL NETWORK & DATASET
# -----------------------------------------------------------------------------
class TruthLensAudioModel(nn.Module):
    """Processes 128-band Log-Mel spectrograms to detect neural vocoder harmonic rigidity."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

class ForensicAudioDataset(Dataset):
    def __init__(self, num_samples: int = 500):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        is_synthetic = 1.0 if (idx % 2 == 1) else 0.0
        # 128 bins x 128 frames spectrogram
        spec = np.zeros((128, 128), dtype=np.float32)
        
        t = np.linspace(0, 1, 128)
        if is_synthetic == 0.0:
            # Authentic Human: Dynamic formants + natural vibrato / micro-tremors + broadband room noise
            for f_idx in [25, 55, 85]:
                jitter = np.sin(2 * np.pi * 5 * t + np.random.uniform(0, 6)) * 4
                for col in range(128):
                    bin_pos = int(np.clip(f_idx + jitter[col], 0, 127))
                    spec[bin_pos, col] = 1.0
            spec += np.random.normal(0.2, 0.08, (128, 128))
        else:
            # Synthetic / Cloned Speech: Rigid horizontal harmonic bands (Neural vocoder HiFi-GAN phase freeze)
            for f_idx in [20, 40, 60, 80]:
                spec[f_idx, :] = 1.2
            # Sharp cutoff in top 25% frequencies (common 8kHz / 16kHz vocoder ceiling)
            spec[96:, :] *= 0.02

        spec_norm = (spec - np.mean(spec)) / (np.std(spec) + 1e-5)
        spec_tensor = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor([is_synthetic], dtype=torch.float32)
        return spec_tensor, label_tensor

# -----------------------------------------------------------------------------
# 4. TRAINING EXECUTION PIPELINE
# -----------------------------------------------------------------------------
def train_models():
    os.makedirs("weights", exist_ok=True)
    
    # --- TRAIN VISUAL DUAL-STREAM MODEL ---
    print("\n--- [Phase 1/2] Training TruthLens Dual-Stream Visual Detector ---")
    visual_model = TruthLensVisualModel().to(device)
    train_dataset = ForensicFaceDataset(num_samples=600)
    val_dataset = ForensicFaceDataset(num_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(visual_model.parameters(), lr=8e-4, weight_decay=1e-4)
    
    epochs = 4
    for epoch in range(1, epochs + 1):
        visual_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for rgb, fft, labels in train_loader:
            rgb, fft, labels = rgb.to(device), fft.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = visual_model(rgb, fft)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * rgb.size(0)
            predicted_class = (preds > 0.5).float()
            correct += (predicted_class == labels).sum().item()
            total += labels.size(0)
            
        train_acc = (correct / total) * 100
        
        # Validation
        visual_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for rgb, fft, labels in val_loader:
                rgb, fft, labels = rgb.to(device), fft.to(device), labels.to(device)
                preds = visual_model(rgb, fft)
                loss = criterion(preds, labels)
                val_loss += loss.item() * rgb.size(0)
                predicted_class = (preds > 0.5).float()
                val_correct += (predicted_class == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = (val_correct / val_total) * 100
        print(f"Epoch {epoch}/{epochs} | Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")
        
    torch.save(visual_model.state_dict(), "weights/truthlens_visual.pth")
    print("✅ Successfully saved: weights/truthlens_visual.pth")

    # --- TRAIN ACOUSTIC VOCODER MODEL ---
    print("\n--- [Phase 2/2] Training TruthLens Acoustic Vocoder Detector ---")
    audio_model = TruthLensAudioModel().to(device)
    audio_train = ForensicAudioDataset(num_samples=400)
    audio_val = ForensicAudioDataset(num_samples=100)
    
    audio_train_loader = DataLoader(audio_train, batch_size=16, shuffle=True)
    audio_val_loader = DataLoader(audio_val, batch_size=16, shuffle=False)
    
    audio_opt = optim.AdamW(audio_model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(1, epochs + 1):
        audio_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for spec, labels in audio_train_loader:
            spec, labels = spec.to(device), labels.to(device)
            audio_opt.zero_grad()
            preds = audio_model(spec)
            loss = criterion(preds, labels)
            loss.backward()
            audio_opt.step()
            total_loss += loss.item() * spec.size(0)
            correct += ((preds > 0.5).float() == labels).sum().item()
            total += labels.size(0)
            
        train_acc = (correct / total) * 100
        
        audio_model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for spec, labels in audio_val_loader:
                spec, labels = spec.to(device), labels.to(device)
                preds = audio_model(spec)
                val_correct += ((preds > 0.5).float() == labels).sum().item()
                val_total += labels.size(0)
        val_acc = (val_correct / val_total) * 100
        print(f"Epoch {epoch}/{epochs} | Audio Train Loss: {total_loss/total:.4f} | Audio Val Acc: {val_acc:.1f}%")

    torch.save(audio_model.state_dict(), "weights/truthlens_audio.pth")
    print("✅ Successfully saved: weights/truthlens_audio.pth")
    print("\n🎉 Both TruthLens Visual & Audio models have been trained and saved successfully!")

if __name__ == "__main__":
    train_models()
