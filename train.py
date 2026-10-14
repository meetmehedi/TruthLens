"""
TruthLens – Full Training Pipeline
Visual model: MobileNetV2 spatial + SpectralCNN + HF-power feature, trained with
augmentation, focal loss, cosine LR schedule, and 15 epochs.
Audio model: ConvNet on log-mel spectrograms, 10 epochs.
"""
import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Training TruthLens using hardware: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. VISUAL DUAL-STREAM NETWORK  (must match src/models/visual.py)
# ─────────────────────────────────────────────────────────────────────────────
class SpectralCNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )

    def forward(self, x):
        return self.conv(x)


class TruthLensVisualModel(nn.Module):
    """
    Dual-stream: MobileNetV2 spatial + SpectralCNN (2D-FFT) + HF-power scalar.
    """
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Unfreeze last 4 feature blocks for fine-tuning
        for name, param in mobilenet.features.named_parameters():
            block_idx = int(name.split('.')[0]) if name[0].isdigit() else -1
            param.requires_grad = block_idx >= 14
        self.spatial_backbone = mobilenet.features
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.spectral_branch = SpectralCNNBranch()
        # +1 for hf_power scalar
        self.fusion_head = nn.Sequential(
            nn.Linear(1280 + 128 + 1, 512),
            nn.LayerNorm(512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.LayerNorm(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_tensor, fft_tensor, hf_power_scalar):
        spatial_feats = self.spatial_pool(self.spatial_backbone(rgb_tensor)).flatten(1)
        spectral_feats = self.spectral_branch(fft_tensor)
        combined = torch.cat([spatial_feats, spectral_feats, hf_power_scalar], dim=1)
        return self.fusion_head(combined)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FOCAL LOSS  (penalises easy confident predictions, focuses on hard ones)
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce = nn.functional.binary_cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 3. DISK-BACKED DATASET  with heavy augmentation
# ─────────────────────────────────────────────────────────────────────────────
def _hf_power(gray: np.ndarray) -> float:
    """High-frequency power ratio — SDXL/diffusion ~0.09, real cameras ~0.20+."""
    h, w = gray.shape
    dft = np.fft.fftshift(np.fft.fft2(gray.astype(float)))
    mag = np.abs(dft)
    cy, cx = h // 2, w // 2
    yi, xi = np.ogrid[:h, :w]
    dist = np.sqrt((xi - cx) ** 2 + (yi - cy) ** 2)
    hf_mask = dist > min(cy, cx) * 0.7
    return float(np.sum(mag[hf_mask]) / (np.sum(mag) + 1e-9))


class DiskForensicFaceDataset(Dataset):
    """
    Loads real + synthetic faces from data/.
    Training: heavy augmentation (flips, colour jitter, blur, random erase).
    Validation: no augmentation.
    """
    def __init__(self, data_root: str = "data", split: str = "train",
                 split_ratio: float = 0.80):
        self.split = split
        self.samples = []

        real_files = sorted(glob.glob(os.path.join(data_root, "real_faces", "*.jpg")))
        fake_files = sorted(glob.glob(os.path.join(data_root, "fake_faces", "*.jpg")))

        split_real = int(len(real_files) * split_ratio)
        split_fake = int(len(fake_files) * split_ratio)

        sel_real = real_files[:split_real] if split == "train" else real_files[split_real:]
        sel_fake = fake_files[:split_fake] if split == "train" else fake_files[split_fake:]

        for p in sel_real: self.samples.append((p, 0.0))
        for p in sel_fake: self.samples.append((p, 1.0))
        random.shuffle(self.samples)

        # Base transforms (always applied after numpy→tensor)
        self.base_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # Extra augmentation for training
        self.aug_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.10)),
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

        # Spatial tensor
        rgb_t = self.base_tf(rgb)
        if self.split == "train":
            rgb_t = self.aug_tf(rgb_t)

        # FFT spectrum tensor
        gray112 = cv2.cvtColor(cv2.resize(bgr, (112, 112)), cv2.COLOR_BGR2GRAY)
        f = np.fft.fftshift(np.fft.fft2(gray112.astype(np.float32)))
        mag = 20 * np.log(np.abs(f) + 1e-6)
        mag_norm = (mag - np.mean(mag)) / (np.std(mag) + 1e-5)
        fft_t = torch.tensor(mag_norm, dtype=torch.float32).unsqueeze(0)

        # HF power scalar (key AI-vs-real discriminator)
        gray224 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hf = _hf_power(gray224)
        hf_t = torch.tensor([[hf]], dtype=torch.float32).squeeze(0)  # shape [1]

        label_t = torch.tensor([label], dtype=torch.float32)
        return rgb_t, fft_t, hf_t, label_t

    def get_weights(self):
        """Returns per-sample weights for balanced sampling."""
        labels = [lbl for _, lbl in self.samples]
        n_real = labels.count(0.0)
        n_fake = labels.count(1.0)
        w_real = 1.0 / n_real if n_real else 1.0
        w_fake = 1.0 / n_fake if n_fake else 1.0
        return [w_real if l == 0.0 else w_fake for l in labels]


# ─────────────────────────────────────────────────────────────────────────────
# 4. AUDIO MODEL  (unchanged architecture)
# ─────────────────────────────────────────────────────────────────────────────
class TruthLensAudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1), nn.Sigmoid()
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
        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        import scipy.io.wavfile as wav
        p, label = self.samples[idx]
        sr, sig = wav.read(p)
        sig = sig.astype(np.float32) / 32768.0
        if len(sig) < 16384:
            sig = np.pad(sig, (0, 16384 - len(sig)))
        else:
            sig = sig[:16384]
        spec = np.abs(np.fft.rfft(sig.reshape(64, 256), axis=-1))
        spec = cv2.resize(spec, (128, 128))
        spec_norm = (spec - np.mean(spec)) / (np.std(spec) + 1e-5)
        return torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0), \
               torch.tensor([label], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_visual(epochs: int = 15):
    print("\n" + "="*60)
    print("  [Phase 1/2] Visual Detector — 15 epoch training")
    print("="*60)

    train_ds = DiskForensicFaceDataset(split="train")
    val_ds   = DiskForensicFaceDataset(split="val")
    print(f"Dataset  ▸ {len(train_ds)} train  |  {len(val_ds)} val")

    # Balanced sampling for skewed datasets
    weights  = train_ds.get_weights()
    sampler  = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False,    num_workers=0)

    model     = TruthLensVisualModel().to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for rgb, fft, hf, labels in train_loader:
            rgb, fft, hf, labels = rgb.to(device), fft.to(device), hf.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(rgb, fft, hf)
            loss  = criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += ((preds >= 0.5) == (labels >= 0.5)).sum().item()
            total      += len(labels)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for rgb, fft, hf, labels in val_loader:
                rgb, fft, hf, labels = rgb.to(device), fft.to(device), hf.to(device), labels.to(device)
                preds = model(rgb, fft, hf)
                vc += ((preds >= 0.5) == (labels >= 0.5)).sum().item()
                vt += len(labels)

        val_acc  = vc / max(vt, 1) * 100
        train_acc = correct / total * 100
        lr_now   = scheduler.get_last_lr()[0]
        scheduler.step()

        print(f"  Epoch [{epoch:2d}/{epochs}]  "
              f"loss={total_loss/total:.4f}  "
              f"train={train_acc:.1f}%  val={val_acc:.1f}%  "
              f"lr={lr_now:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    # Save best checkpoint
    os.makedirs("weights", exist_ok=True)
    torch.save(best_state, "weights/truthlens_visual.pth")
    print(f"\n✅ Best val acc {best_val_acc:.1f}%  →  saved weights/truthlens_visual.pth")


def train_audio(epochs: int = 10):
    print("\n" + "="*60)
    print("  [Phase 2/2] Audio Detector — 10 epoch training")
    print("="*60)

    train_ds = DiskAudioDataset(split="train")
    val_ds   = DiskAudioDataset(split="val")
    print(f"Dataset  ▸ {len(train_ds)} train  |  {len(val_ds)} val")

    if len(train_ds) == 0:
        print("  ⚠️  No audio data found — skipping audio training.")
        return

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False)

    model     = TruthLensAudioModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    best_val = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for spec, labels in train_loader:
            spec, labels = spec.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(spec)
            loss  = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += ((preds >= 0.5) == (labels >= 0.5)).sum().item()
            total      += len(labels)

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for spec, labels in val_loader:
                spec, labels = spec.to(device), labels.to(device)
                preds = model(spec)
                vc += ((preds >= 0.5) == (labels >= 0.5)).sum().item()
                vt += len(labels)

        val_acc = vc / max(vt, 1) * 100
        scheduler.step()
        print(f"  Epoch [{epoch:2d}/{epochs}]  loss={total_loss/total:.4f}  "
              f"train={correct/total*100:.1f}%  val={val_acc:.1f}%")

        if val_acc >= best_val:
            best_val   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    torch.save(best_state, "weights/truthlens_audio.pth")
    print(f"\n✅ Best val acc {best_val:.1f}%  →  saved weights/truthlens_audio.pth")


if __name__ == "__main__":
    train_visual(epochs=15)
    train_audio(epochs=10)
    print("\n🎉 All models trained and saved!")
