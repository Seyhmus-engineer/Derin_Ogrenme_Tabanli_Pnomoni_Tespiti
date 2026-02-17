import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm

from PIL import Image
import cv2


# =========================================================
# 1) CONFIG
# =========================================================
CONFIG = {
    "data_path": r"C:\Python Projeler\Akciger_zatürre\data",
    # Klinik pipeline: modelin yeri burası olsun (senin gradcam tarafıyla uyumlu)
    "save_path": r"C:\Python Projeler\Akciger_zatürre\data\models",

    "img_size": 224,
    "batch_size": 8,
    "num_workers": 2,

    "epochs": 40,
    "patience": 6,

    "lr": 2e-4,          # AdamW ile iyi
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,

    # Warmup + Cosine
    "warmup_ratio": 0.10,
    "min_lr_ratio": 0.05,

    # Klinik amaçlı anti-shortcut ayarları
    "use_roi_crop": True,      # Akciğer ROI'ye yaklaştırma
    "use_clahe": True,         # Kontrast standardizasyonu (özellikle CXR için)
    "remove_border": True,     # Kenar/padding etkisini kırmak için crop
    "border_crop_ratio": 0.04, # %4 kırp

    # AMP
    "use_amp": True,

    # Repro
    "seed": 42,
}


# =========================================================
# 2) HELPERS
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_div(a, b):
    return a / b if b != 0 else 0.0


@torch.no_grad()
def confusion_counts(logits, y_true, pos_label_idx=1):
    preds = torch.argmax(logits, dim=1)
    tp = ((preds == pos_label_idx) & (y_true == pos_label_idx)).sum().item()
    fp = ((preds == pos_label_idx) & (y_true != pos_label_idx)).sum().item()
    tn = ((preds != pos_label_idx) & (y_true != pos_label_idx)).sum().item()
    fn = ((preds != pos_label_idx) & (y_true == pos_label_idx)).sum().item()
    return tp, fp, tn, fn


def prf_from_counts(tp, fp, tn, fn):
    precision = safe_div(tp, (tp + fp))
    recall = safe_div(tp, (tp + fn))
    f1 = safe_div(2 * precision * recall, (precision + recall))
    acc = safe_div((tp + tn), (tp + tn + fp + fn))
    return acc, precision, recall, f1


def make_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# =========================================================
# 3) CLINIC PREPROCESS: ROI CROP + CLAHE + BORDER CROP
# =========================================================
def border_crop(gray_u8: np.ndarray, ratio: float = 0.04) -> np.ndarray:
    if ratio <= 0:
        return gray_u8
    h, w = gray_u8.shape[:2]
    dh = int(h * ratio)
    dw = int(w * ratio)
    if h - 2 * dh <= 10 or w - 2 * dw <= 10:
        return gray_u8
    return gray_u8[dh:h - dh, dw:w - dw]


def clahe_enhance(gray_u8: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_u8)


def auto_roi_crop(gray_u8: np.ndarray) -> np.ndarray:
    """
    Heuristic ROI crop:
    - amaç: padding/etiket/kenar etkisini azaltıp toraks bölgesine yaklaşmak
    - bu segmentation değil; ama shortcut'ı ciddi kırar
    """
    img = gray_u8.copy()

    # normalize/contrast
    img_eq = cv2.equalizeHist(img)

    # adaptif threshold -> gövde maskesi benzeri
    th = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 51, 2)
    th = 255 - th

    kernel = np.ones((9, 9), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray_u8

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # çok küçükse crop yapma
    if w * h < 0.25 * gray_u8.shape[0] * gray_u8.shape[1]:
        return gray_u8

    pad = int(0.04 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray_u8.shape[1], x + w + pad)
    y1 = min(gray_u8.shape[0], y + h + pad)

    cropped = gray_u8[y0:y1, x0:x1]
    return cropped


def preprocess_pil_to_gray_np(pil_img: Image.Image) -> np.ndarray:
    gray = np.array(pil_img.convert("L")).astype(np.uint8)
    return gray


def gray_np_to_pil_rgb(gray_u8: np.ndarray) -> Image.Image:
    # EfficientNet ImageNet mean/std beklediği için 3 kanal veriyoruz
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    return Image.fromarray(rgb)


# =========================================================
# 4) DATASET
# =========================================================
class ImageFolderClinic(torch.utils.data.Dataset):
    """
    Klasik ImageFolder yerine:
    - Görseli PIL ile okur
    - ROI crop + border crop + CLAHE uygular
    - Sonra transform
    """
    def __init__(self, root_dir: str, transform=None, use_roi=True, use_clahe_=True,
                 remove_border_=True, border_ratio=0.04):
        self.root = Path(root_dir)
        self.transform = transform
        self.use_roi = use_roi
        self.use_clahe = use_clahe_
        self.remove_border = remove_border_
        self.border_ratio = border_ratio

        # ImageFolder benzeri: class klasörleri
        classes = [d.name for d in self.root.iterdir() if d.is_dir()]
        classes = sorted(classes)
        if len(classes) == 0:
            raise FileNotFoundError(f"Sınıf klasörü yok: {root_dir}")

        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        exts = (".jpg", ".jpeg", ".png")
        self.samples = []
        for c in classes:
            cdir = self.root / c
            for p in cdir.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    self.samples.append((str(p), self.class_to_idx[c]))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"Görsel bulunamadı: {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y = self.samples[idx]
        pil = Image.open(img_path)

        gray = preprocess_pil_to_gray_np(pil)

        if self.remove_border:
            gray = border_crop(gray, ratio=self.border_ratio)
        if self.use_roi:
            gray = auto_roi_crop(gray)
        if self.use_clahe:
            gray = clahe_enhance(gray)

        pil_rgb = gray_np_to_pil_rgb(gray)

        if self.transform:
            x = self.transform(pil_rgb)
        else:
            x = transforms.ToTensor()(pil_rgb)

        return x, y


def build_transforms(img_size=224):
    """
    Klinik amaç: shortcut kırmak için güçlü augment
    """
    train_tfms = transforms.Compose([
        # border ve padding bağımlılığını kırar
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.90, 1.10)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=0.0),

        # kontrast/brightness varyasyonu
        transforms.ColorJitter(brightness=0.15, contrast=0.20),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),

        # lokal “silme”: modelin tek noktaya yapışmasını azaltır
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3), value="random"),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms


def get_dataloaders(data_dir: str, batch_size: int, num_workers: int):
    data_dir = Path(data_dir)

    train_tfms, eval_tfms = build_transforms(CONFIG["img_size"])

    train_root = data_dir / "train"
    val_root = data_dir / "val"
    test_root = data_dir / "test"

    if not train_root.exists():
        raise FileNotFoundError(f"Klasör bulunamadı: {train_root}")

    train_ds = ImageFolderClinic(
        root_dir=str(train_root),
        transform=train_tfms,
        use_roi=CONFIG["use_roi_crop"],
        use_clahe_=CONFIG["use_clahe"],
        remove_border_=CONFIG["remove_border"],
        border_ratio=CONFIG["border_crop_ratio"],
    )
    val_ds = ImageFolderClinic(
        root_dir=str(val_root),
        transform=eval_tfms,
        use_roi=CONFIG["use_roi_crop"],
        use_clahe_=CONFIG["use_clahe"],
        remove_border_=CONFIG["remove_border"],
        border_ratio=CONFIG["border_crop_ratio"],
    )
    test_ds = ImageFolderClinic(
        root_dir=str(test_root),
        transform=eval_tfms,
        use_roi=CONFIG["use_roi_crop"],
        use_clahe_=CONFIG["use_clahe"],
        remove_border_=CONFIG["remove_border"],
        border_ratio=CONFIG["border_crop_ratio"],
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.class_to_idx, train_ds


# =========================================================
# 5) MODEL
# =========================================================
def build_efficientnet_b0(num_classes: int = 2):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def compute_class_weights(train_ds, class_to_idx):
    """
    Imbalance için class weight:
    - CE loss -> weight=[w0, w1]
    """
    counts = np.zeros(len(class_to_idx), dtype=np.int64)
    for _, y in train_ds.samples:
        counts[y] += 1
    counts = np.maximum(counts, 1)
    inv = 1.0 / counts.astype(np.float32)
    w = inv / inv.sum() * len(inv)
    return torch.tensor(w, dtype=torch.float32), counts


# =========================================================
# 6) TRAIN/VAL LOOP
# =========================================================
def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None, pos_idx=1):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches = 0
    TP = FP = TN = FN = 0

    pbar = tqdm(loader, desc=("Train" if is_train else "Eval"), leave=False, mininterval=0.5)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if is_train and scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler:
                    scheduler.step()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()

        total_loss += float(loss.item())
        n_batches += 1

        tp, fp, tn, fn = confusion_counts(logits, y, pos_label_idx=pos_idx)
        TP += tp; FP += fp; TN += tn; FN += fn

    avg_loss = total_loss / max(1, n_batches)
    acc, precision, recall, f1 = prf_from_counts(TP, FP, TN, FN)
    return avg_loss, acc, precision, recall, f1


# =========================================================
# 7) MAIN
# =========================================================
def main():
    set_seed(CONFIG["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n🚀 GPU Bulundu: {torch.cuda.get_device_name(0)}")
    else:
        print("\n⚠️ GPU Bulunamadı! CPU ile çalışıyor.")

    os.makedirs(CONFIG["save_path"], exist_ok=True)
    ckpt_path = os.path.join(CONFIG["save_path"], "best_model.pth")

    print(f"📂 Veriler okunuyor: {CONFIG['data_path']}")
    train_loader, val_loader, test_loader, class_to_idx, train_ds = get_dataloaders(
        data_dir=CONFIG["data_path"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )
    print(f"✅ Sınıflar: {class_to_idx}")
    pos_idx = class_to_idx.get("PNEUMONIA", 1)
    print(f"ℹ️ Pozitif sınıf index: {pos_idx}")

    class_w, counts = compute_class_weights(train_ds, class_to_idx)
    print(f"📊 Train sınıf sayıları: {counts.tolist()} | class_weight: {class_w.tolist()}")

    model = build_efficientnet_b0(num_classes=2).to(device)

    # Weighted CE + label smoothing
    criterion = nn.CrossEntropyLoss(
        weight=class_w.to(device),
        label_smoothing=CONFIG["label_smoothing"]
    )

    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * CONFIG["epochs"]
    warmup_steps = int(CONFIG["warmup_ratio"] * total_steps)
    scheduler = make_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, CONFIG["min_lr_ratio"])

    scaler = torch.cuda.amp.GradScaler() if (device == "cuda" and CONFIG["use_amp"]) else None

    best_val_f1 = 0.0
    best_val_loss = float("inf")
    early_stop_counter = 0

    print(f"\n🏁 Eğitim başlıyor (epochs={CONFIG['epochs']})")
    print("-" * 70)

    for epoch in range(1, CONFIG["epochs"] + 1):
        tr_loss, tr_acc, tr_p, tr_r, tr_f1 = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            pos_idx=pos_idx
        )

        va_loss, va_acc, va_p, va_r, va_f1 = run_epoch(
            model, val_loader, criterion, device,
            optimizer=None, scheduler=None, scaler=None,
            pos_idx=pos_idx
        )

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | LR: {lr_now:.2e}")
        print(f"  Train -> Loss: {tr_loss:.4f} | F1: {tr_f1:.4f} | Recall: {tr_r:.4f}")
        print(f"  Val   -> Loss: {va_loss:.4f} | F1: {va_f1:.4f} | Recall: {va_r:.4f}")

        # save best by F1
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "epoch": epoch,
                "val_f1": best_val_f1,
                "config": CONFIG,
            }, ckpt_path)
            print(f"  💾 Best model saved -> {ckpt_path}")

        # early stopping by val loss
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"  ⚠️ EarlyStop: {early_stop_counter}/{CONFIG['patience']}")
            if early_stop_counter >= CONFIG["patience"]:
                print(f"\n🛑 Early stopping: val loss {CONFIG['patience']} epoch düşmedi.")
                break

        print("-" * 70)

    # TEST
    print("\n✅ Eğitim bitti, test aşaması...")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        print("✅ En iyi model yüklendi.")

    te_loss, te_acc, te_p, te_r, te_f1 = run_epoch(
        model, test_loader, criterion, device,
        optimizer=None, scheduler=None, scaler=None,
        pos_idx=pos_idx
    )

    print("\n" + "=" * 34)
    print("🏆 NİHAİ TEST SONUÇLARI (CLINIC)")
    print("=" * 34)
    print(f"Accuracy  : {te_acc:.4f}")
    print(f"Recall    : {te_r:.4f}  (kritik)")
    print(f"Precision : {te_p:.4f}")
    print(f"F1        : {te_f1:.4f}")
    print("=" * 34)


if __name__ == "__main__":
    main()
