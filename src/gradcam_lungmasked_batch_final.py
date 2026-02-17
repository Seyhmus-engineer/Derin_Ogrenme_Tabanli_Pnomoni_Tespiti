import os
import re
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import cv2

# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    # Test klasörün (ImageFolder yapısı şart değil; alt klasörlerde de arar)
    "data_test_dir": r"C:\Python Projeler\Akciger_zatürre\data\test",

    # ✅ Yeni model yolun (senin attığın)
    "ckpt_path": r"C:\Python Projeler\Akciger_zatürre\data\models\best_clinical_model.pth",

    # Çıktı klasörü
    "out_dir": r"C:\Python Projeler\Akciger_zatürre\outputs\lungmasked",

    # Kaç görsel üretilecek
    "num_images": 6,
    "seed": 42,

    # Classifier input
    "img_size": 224,

    # Segmentation çalışma boyutu
    "seg_size": 512,

    # Overlay ayarları
    "alpha": 0.40,  # overlay saydamlığı
    "percentile_clip": 97.0,  # 95-99 arası iyi çalışır
    "smooth_kernel": 31,  # 0/None kapatır (224 üstünde blur mantıklı)

    # Akciğer maskesi ayarları
    "mask_thr": 0.35,  # 0.25-0.45 arası denenir
    "mask_erode": 0,  # 0/1/2 (akciğer dışına taşmayı azaltır)
    "mask_blur": 9,  # 0 kapatır; kenarı yumuşatır

    # İstersen sadece akciğer bbox’ına crop (classifier için)
    "use_bbox_crop": True,
    "bbox_pad": 0.08,

    # Kaydetme panel boyutu
    "panel_header_h": 60,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# UTILS
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str, maxlen: int = 160) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:maxlen]


def list_images(root: str):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    rootp = Path(root)
    # rglob ile tüm alt klasörleri tarar (NORMAL, PNEUMONIA dahil)
    return [p for p in rootp.rglob("*") if p.suffix.lower() in exts]


def normalize_0_1(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    return x / (x.max() + eps)


def clip_heatmap_percentile(hm: np.ndarray, p: float) -> np.ndarray:
    p = float(p)
    p = max(0.0, min(100.0, p))
    th = np.percentile(hm, p)
    hm = hm / (th + 1e-8)
    return np.clip(hm, 0, 1)


def smooth_heatmap_224(hm_224: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1:
        return hm_224
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(hm_224, (k, k), 0)


def make_overlay_from_gray(gray_uint8: np.ndarray, heat_0_1: np.ndarray, alpha: float):
    gray_bgr = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
    heat_uint8 = (np.clip(heat_0_1, 0, 1) * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(gray_bgr, 1.0 - alpha, heat_bgr, alpha, 0)
    return gray_bgr, heat_bgr, overlay


def save_triplet_panel(orig_bgr, heat_bgr, overlay_bgr, title_text: str, out_path: str, header_h: int):
    h, w, _ = orig_bgr.shape
    panel = np.ones((h + header_h, w * 3, 3), dtype=np.uint8) * 255

    panel[header_h:header_h + h, 0:w] = orig_bgr
    panel[header_h:header_h + h, w:2 * w] = heat_bgr
    panel[header_h:header_h + h, 2 * w:3 * w] = overlay_bgr

    # Title
    cv2.putText(panel, title_text, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 15, 15), 2, cv2.LINE_AA)
    # Labels
    cv2.putText(panel, "Original", (15, header_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (15, 15, 15), 2, cv2.LINE_AA)
    cv2.putText(panel, "Grad-CAM", (w + 15, header_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (15, 15, 15), 2, cv2.LINE_AA)
    cv2.putText(panel, "Overlay", (2 * w + 15, header_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (15, 15, 15), 2,
                cv2.LINE_AA)

    safe_mkdir(os.path.dirname(out_path))

    ok = cv2.imwrite(out_path, panel)
    if not ok:
        ok2, buf = cv2.imencode(".png", panel)
        if not ok2:
            raise RuntimeError(f"Kayıt başarısız: {out_path}")
        with open(out_path, "wb") as f:
            f.write(buf.tobytes())


# =========================================================
# CLASSIFIER
# =========================================================
def build_efficientnet_b0(num_classes: int = 2):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_classifier_ckpt(model: nn.Module, ckpt_path: str, device: str):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # Hata yönetimi: model yapısı uyuşmazsa strict=False ile yüklemeyi dene
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"⚠️ Uyarı: Strict yükleme başarısız oldu, esnek yükleme deneniyor... ({e})")
        model.load_state_dict(state, strict=False)

    class_to_idx = ckpt.get("class_to_idx", None) if isinstance(ckpt, dict) else None
    return model, class_to_idx


# =========================================================
# SEGMENTATION (torchxrayvision PSPNet)
# =========================================================
def load_lung_seg_model(device: str):
    try:
        import torchxrayvision as xrv
    except ImportError:
        print("❌ HATA: torchxrayvision kütüphanesi yüklü değil!")
        print("Lütfen terminale şunu yazıp kur: pip install torchxrayvision")
        sys.exit(1)

    seg_model = xrv.baseline_models.chestx_det.PSPNet().to(device)
    seg_model.eval()
    seg_model.transform = lambda x: x
    return seg_model


def prepare_seg_input(pil_img: Image.Image, seg_size: int) -> torch.Tensor:
    # 1 kanal (Grayscale)
    img = pil_img.convert("L").resize((seg_size, seg_size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    # XRV norm: -1024..1024
    arr = (arr / 255.0) * 2048.0 - 1024.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return x


@torch.no_grad()
def get_lung_mask(seg_model, pil_img: Image.Image, seg_size: int, thr: float,
                  erode_iter: int = 0, blur_k: int = 0) -> np.ndarray:
    x = prepare_seg_input(pil_img, seg_size=seg_size).to(device)  # [1,1,H,W]
    out = seg_model(x)

    targets = getattr(seg_model, "targets", None)
    if targets and ("Left Lung" in targets) and ("Right Lung" in targets):
        li = targets.index("Left Lung")
        ri = targets.index("Right Lung")
        lung = out[0, li] + out[0, ri]
    else:
        lung = out[0, 0] + (out[0, 1] if out.size(1) > 1 else 0)

    lung = lung.detach().float().cpu().numpy()
    lung = normalize_0_1(lung)

    mask = (lung > float(thr)).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    if erode_iter and erode_iter > 0:
        mask = cv2.erode(mask, k, iterations=int(erode_iter))

    return mask


def mask_to_bbox(mask01: np.ndarray, pad_ratio: float):
    ys, xs = np.where(mask01 > 0.1)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    pad_y = int(h * pad_ratio)
    pad_x = int(w * pad_ratio)
    y1 = max(0, y1 - pad_y)
    y2 = min(mask01.shape[0] - 1, y2 + pad_y)
    x1 = max(0, x1 - pad_x)
    x2 = min(mask01.shape[1] - 1, x2 + pad_x)
    return (x1, y1, x2, y2)


# =========================================================
# GRAD-CAM
# =========================================================
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def fwd_hook(_, __, output):
            self.activations = output

        def bwd_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, x: torch.Tensor, class_idx: int):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=False)

        A = self.activations
        G = self.gradients
        if A is None or G is None:
            raise RuntimeError("GradCAM hook çalışmadı. Modelin layer yapısını kontrol et.")

        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=False)
        cam = torch.relu(cam)[0].detach().cpu().numpy()
        cam = normalize_0_1(cam)
        return cam, logits.detach()


def find_last_spatial_layer(model: nn.Module, input_size: int = 224) -> nn.Module:
    # EfficientNet için genellikle en son features bloğu iş görür
    # model.features[-1]
    return model.features[-1]


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(CONFIG["seed"])
    safe_mkdir(CONFIG["out_dir"])

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(CONFIG["ckpt_path"]):
        print(f"❌ Checkpoint bulunamadı: {CONFIG['ckpt_path']}")
        return

    # 1. Modeli Yükle
    clf = build_efficientnet_b0(num_classes=2).to(device)
    clf, class_to_idx = load_classifier_ckpt(clf, CONFIG["ckpt_path"], device)
    clf.eval()

    if class_to_idx is None:
        class_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}
    pneu_idx = class_to_idx.get("PNEUMONIA", 1)
    inv_map = {v: k for k, v in class_to_idx.items()}

    # 2. Segmentasyon Modelini Yükle
    print("Segmentasyon modeli yükleniyor (torchxrayvision)...")
    seg_model = load_lung_seg_model(device)
    print("✅ Segmentasyon modeli hazır.")

    # 3. Grad-CAM Hazırlığı
    target_layer = find_last_spatial_layer(clf, input_size=CONFIG["img_size"])
    cam_engine = GradCAM(clf, target_layer)

    cls_tfm = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. Resimleri Bul
    files = list_images(CONFIG["data_test_dir"])
    if len(files) == 0:
        print(f"❌ Test klasöründe resim bulunamadı: {CONFIG['data_test_dir']}")
        return

    n = min(int(CONFIG["num_images"]), len(files))
    picks = random.sample(files, n)
    print(f"Toplam {len(files)} resim bulundu. {n} tanesi işleniyor...")

    # 5. İşleme Döngüsü (DÜZELTİLMİŞ & SIKILAŞTIRILMIŞ)
    for i, path in enumerate(tqdm(picks, total=n)):
        try:
            pil_rgb = Image.open(path).convert("RGB")

            # --- ADIM 1: Maskeyi Al ---
            # Maskeyi biraz daha sert (thr=0.40) alalım ki gürültü azalsın
            mask = get_lung_mask(seg_model, pil_rgb, seg_size=CONFIG["seg_size"], thr=0.40)

            # Seg boyutuna resize
            img_seg_rgb = np.array(pil_rgb.resize((CONFIG["seg_size"], CONFIG["seg_size"]), Image.BILINEAR),
                                   dtype=np.uint8)

            # --- ADIM 2: Maskeyi "Sıkılaştır" (Klinik Hile) ---
            # Kenarlardan içeri gir (Erosion) -> Kaburgaları atar
            kernel = np.ones((15, 15), np.uint8)  # Kernel boyutunu büyüttük
            strict_mask = cv2.erode(mask, kernel, iterations=3)

            # Üst kısmı traşla (Boyun/Omuz shortcut'ını öldür)
            h_mask, w_mask = strict_mask.shape
            top_cut = int(h_mask * 0.15)  # Üstten %15'i kes
            strict_mask[:top_cut, :] = 0

            # --- ADIM 3: Classifier Input ---
            # Classifier'a yine de orijinal maskeli halini verelim (Feature kaybetmemek için)
            bbox = mask_to_bbox(mask, pad_ratio=CONFIG["bbox_pad"]) if CONFIG["use_bbox_crop"] else None

            if bbox:
                x1, y1, x2, y2 = bbox
                crop_rgb = img_seg_rgb[y1:y2 + 1, x1:x2 + 1].copy()
                crop_mask = strict_mask[y1:y2 + 1, x1:x2 + 1].copy()  # Buraya strict mask verdim
            else:
                crop_rgb = img_seg_rgb.copy()
                crop_mask = strict_mask.copy()

            # Input hazırlığı
            crop_rgb_for_clf = crop_rgb.copy()
            crop_rgb_for_clf[crop_mask == 0] = 0

            pil_clf = Image.fromarray(crop_rgb_for_clf)
            x = cls_tfm(pil_clf).unsqueeze(0).to(device)

            # --- ADIM 4: Tahmin ---
            with torch.no_grad():
                logits = clf(x)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                pred_idx = int(np.argmax(probs))
                pred_name = inv_map.get(pred_idx, str(pred_idx))
                p_pneu = float(probs[pneu_idx])

            # --- ADIM 5: Grad-CAM ---
            heat_small, _ = cam_engine(x, class_idx=pneu_idx)

            # --- ADIM 6: Görselleştirme (Sıkı Maske Uygulaması) ---
            # Isı haritasını büyüt
            heat_224 = cv2.resize(heat_small, (CONFIG["img_size"], CONFIG["img_size"]), interpolation=cv2.INTER_CUBIC)

            # Maskeyi de 224 yap (Sıkı maskeyi kullanıyoruz!)
            mask_224 = cv2.resize(crop_mask.astype(np.float32), (CONFIG["img_size"], CONFIG["img_size"]),
                                  interpolation=cv2.INTER_NEAREST)

            # KRİTİK NOKTA: Isı haritasını maske ile çarp -> Maske dışındaki ısıyı SIFIRLA
            heat_224 = heat_224 * mask_224

            # Düzenleme
            heat_224 = clip_heatmap_percentile(heat_224, CONFIG["percentile_clip"])
            heat_224 = smooth_heatmap_224(heat_224, CONFIG["smooth_kernel"])

            # Tekrar maske ile çarp (Blur işleminden sonra dışarı taşanı silmek için)
            heat_224 = heat_224 * mask_224
            heat_224 = normalize_0_1(heat_224)

            # Gri tonlama resim
            crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
            crop_gray = cv2.resize(crop_gray, (CONFIG["img_size"], CONFIG["img_size"]), interpolation=cv2.INTER_CUBIC)

            # Overlay oluştur
            orig_bgr, heat_bgr, overlay_bgr = make_overlay_from_gray(crop_gray, heat_224, alpha=CONFIG["alpha"])

            # Kaydet
            base = sanitize_filename(path.name)
            title = f"Pred: {pred_name} | P(PNEU)={p_pneu:.2f}"  # Başlığı kısalttım sığsın diye
            out_name = f"lungcam_{i + 1:02d}_{pred_name}_{base}.png"
            out_path = os.path.join(CONFIG["out_dir"], out_name)

            save_triplet_panel(orig_bgr, heat_bgr, overlay_bgr, title, out_path, int(CONFIG["panel_header_h"]))

        except Exception as e:
            print(f"\n⚠️ Hata oluştu ({path.name}): {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✅ Bitti! Görseller şuraya kaydedildi: {CONFIG['out_dir']}")

if __name__ == "__main__":
    main()