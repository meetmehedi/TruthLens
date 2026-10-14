import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Training TruthLens on physical dataset using hardware: {device}")

# -----------------------------------------------------------------------------
# 1. VISUAL DUAL-STREAM NEURAL NETWORK
# -----------------------------------------------------------------------------
class SpectralCNNBranch(nn.Module):
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
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.spatial_backbone = mobilenet.features
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.spectral_branch = SpectralCNNBranch()
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
# 2. DISK-BACKED PHYSICAL DATASET LOADER
# -----------------------------------------------------------------------------
class DiskForensicFaceDataset(Dataset):
    """
    Loads authentic and synthetic faces directly from the physical `data/` directory.
    Extracts RGB spatial features and 2D-FFT frequency spectra for dual-stream training.
    """
    def __init__(self, data_root: str = "data", split: str = "train", split_ratio: float = 0.8):
        self.samples = []
        real_files = sorted(glob.glob(os.path.join(data_root, "real_faces", "*.jpg")))
        fake_files = sorted(glob.glob(os.path.join(data_root, "fake_faces", "*.jpg")))
        
        split_real = int(len(real_files) * split_ratio)
        split_fake = int(len(fake_files) * split_ratio)
        
        if split == "train":
            selected_real = real_files[:split_real]
            selected_fake = fake_files[:split_fake]
        else:
            selected_real = real_files[split_real:]
            selected_fake = fake_files[split_fake:]
            
        for path in selected_real:
            self.samples.append((path, 0.0))
        for path in selected_fake:
            self.samples.append((path, 1.0))
            
        np.random.shuffle(self.samples)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        bgr = cv2.imread(path)
        if bgr is None:
            bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        bgr = cv2.resize(bgr, (224, 224))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        rgb_tensor = self.transform(rgb)
        
        # Compute 2D-FFT magnitude spectrum
        gray = cv2.cvtColor(cv2.resize(bgr, (112, 112)), cv2.COLOR_BGR2GRAY)
        f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        mag = 20 * np.log(np.abs(f) + 1e-6)
        mag_norm = (mag - np.mean(mag)) / (np.std(mag) + 1e-5)
        fft_tensor = torch.tensor(mag_norm, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        return rgb_tensor, fft_tensor, label_tensor


# -----------------------------------------------------------------------------
# 3. DISK-BACKED AUDIO DATASET LOADER
# -----------------------------------------------------------------------------
class TruthLensAudioModel(nn.Module):
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


class DiskAudioDataset(Dataset):
    def __init__(self, data_root: str = "data", split: str = "train", split_ratio: float = 0.8):
        import scipy.io.wavfile as wav
        self.samples = []
        real_wavs = sorted(glob.glob(os.path.join(data_root, "real_audio", "*.wav")))
        fake_wavs = sorted(glob.glob(os.path.join(data_root, "fake_audio", "*.wav")))
        
        s_r = int(len(real_wavs) * split_ratio)
        s_f = int(len(fake_wavs) * split_ratio)
        
        sel_r = real_wavs[:s_r] if split == "train" else real_wavs[s_r:]
        sel_f = fake_wavs[:s_f] if split == "train" else fake_wavs[s_f:]
        
        for p in sel_r: self.samples.append((p, 0.0))
        for p in sel_f: self.samples.append((p, 1.0))
        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import scipy.io.wavfile as wav
        p, label = self.samples[idx]
        sr, sig = wav.read(p)
        sig = sig.astype(np.float32) / 32768.0
        # Compute STFT spectrogram (128x128)
        if len(sig) < 16384:
            sig = np.pad(sig, (0, 16384 - len(sig)))
        else:
            sig = sig[:16384]
        spec = np.abs(np.fft.rfft(sig.reshape(64, 256), axis=-1))
        spec = cv2.resize(spec, (128, 128))
        spec_norm = (spec - np.mean(spec)) / (np.std(spec) + 1e-5)
        spec_t = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0)
        return spec_t, torch.tensor([label], dtype=torch.float32)


# -----------------------------------------------------------------------------
# 4. TRAINING EXECUTION
# -----------------------------------------------------------------------------
def train_all():
    os.makedirs("weights", exist_ok=True)
    
    print("\n--- [Phase 1/2] Training Visual Detector on Physical Dataset ---")
    train_ds = DiskForensicFaceDataset(split="train")
    val_ds = DiskForensicFaceDataset(split="val")
    print(f"Loaded {len(train_ds)} train samples and {len(val_ds)} validation samples.")
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    visual_model = TruthLensVisualModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(visual_model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(1, 5):
        visual_model.train()
        total_loss, correct, total = 0.0, 0, 0
        for rgb, fft, labels in train_loader:
            rgb, fft, labels = rgb.to(device), fft.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = visual_model(rgb, fft)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += ((preds >= 0.5) == (labels >= 0.5)).sum().item()
            total += len(labels)
            
        visual_model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for rgb, fft, labels in val_loader:
                rgb, fft, labels = rgb.to(device), fft.to(device), labels.to(device)
                preds = visual_model(rgb, fft)
                val_correct += ((preds >= 0.5) == (labels >= 0.5)).sum().item()
                val_total += len(labels)
                
        print(f"Visual Epoch [{epoch}/4] Loss: {total_loss/total:.4f} | Train Acc: {correct/total*100:.1f}% | Val Acc: {val_correct/max(val_total,1)*100:.1f}%")
        
    visual_path = "weights/truthlens_visual.pth"
    torch.save(visual_model.state_dict(), visual_path)
    print(f"✅ Saved trained visual weights to {visual_path}")
    
    print("\n--- [Phase 2/2] Training Audio Detector on Physical Dataset ---")
    train_audio_ds = DiskAudioDataset(split="train")
    val_audio_ds = DiskAudioDataset(split="val")
    print(f"Loaded {len(train_audio_ds)} audio train samples and {len(val_audio_ds)} validation samples.")
    
    audio_train_loader = DataLoader(train_audio_ds, batch_size=8, shuffle=True)
    audio_val_loader = DataLoader(val_audio_ds, batch_size=8, shuffle=False)
    
    audio_model = TruthLensAudioModel().to(device)
    audio_opt = optim.Adam(audio_model.parameters(), lr=2e-3)
    
    for epoch in range(1, 5):
        audio_model.train()
        total_loss, correct, total = 0.0, 0, 0
        for spec, labels in audio_train_loader:
            spec, labels = spec.to(device), labels.to(device)
            audio_opt.zero_grad()
            preds = audio_model(spec)
            loss = criterion(preds, labels)
            loss.backward()
            audio_opt.step()
            total_loss += loss.item() * len(labels)
            correct += ((preds >= 0.5) == (labels >= 0.5)).sum().item()
            total += len(labels)
            
        audio_model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for spec, labels in audio_val_loader:
                spec, labels = spec.to(device), labels.to(device)
                preds = audio_model(spec)
                val_correct += ((preds >= 0.5) == (labels >= 0.5)).sum().item()
                val_total += len(labels)
                
        print(f"Audio Epoch [{epoch}/4] Loss: {total_loss/total:.4f} | Train Acc: {correct/total*100:.1f}% | Val Acc: {val_correct/max(val_total,1)*100:.1f}%")
        
    audio_path = "weights/truthlens_audio.pth"
    torch.save(audio_model.state_dict(), audio_path)
    print(f"✅ Saved trained audio weights to {audio_path}")


if __name__ == "__main__":
    train_all()
