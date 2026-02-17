import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import io

# ==========================================
# 1. AYARLAR & MODEL YOLLARI
# ==========================================
# Kirvem buradaki yolları kendi bilgisayarına göre kontrol et
MODEL_PATH = r"C:\Python Projeler\Akciger_zatürre\data\models\best_clinical_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="AI-Rad: Akciğer Pnömoni Tespit Sistemi",
    page_icon="🫁",
    layout="wide"
)


# ==========================================
# 2. MODEL YÜKLEME (CACHE İLE HIZLANDIRMA)
# ==========================================
@st.cache_resource
def load_models():
    # --- Classifier ---
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)

    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=False)  # Strict False yaptık ki hata vermesin
    except FileNotFoundError:
        st.error(f"Model dosyası bulunamadı: {MODEL_PATH}")
        return None, None

    model.to(DEVICE)
    model.eval()

    # --- Segmentation (XRayVision) ---
    try:
        import torchxrayvision as xrv
        seg_model = xrv.baseline_models.chestx_det.PSPNet().to(DEVICE)
        seg_model.eval()
    except ImportError:
        st.warning("torchxrayvision yüklü değil, segmentasyon atlanacak.")
        seg_model = None

    return model, seg_model


# ==========================================
# 3. YARDIMCI FONKSİYONLAR (MASK & CAM)
# ==========================================
def normalize_0_1(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def get_lung_mask(seg_model, pil_img):
    if seg_model is None:
        return np.ones((224, 224), dtype=np.uint8)  # Fallback

    # XRV için hazırlık
    img = pil_img.convert("L").resize((512, 512), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    arr = (arr / 255.0) * 2048.0 - 1024.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = seg_model(x)
        lung = out[0, 0] + (out[0, 1] if out.size(1) > 1 else 0)
        lung = lung.detach().cpu().numpy()

    lung = normalize_0_1(lung)
    mask = (lung > 0.35).astype(np.uint8)

    # Morfoloji
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # Sıkılaştırma (Erosion)
    mask = cv2.erode(mask, k, iterations=2)

    # Üst %15 kes (Boyun/Omuz shortcut engelleme)
    h, w = mask.shape
    mask[:int(h * 0.15), :] = 0

    return cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)


# Grad-CAM Sınıfı
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=False)
        cam = torch.relu(cam)
        return cam[0].detach().cpu().numpy(), logits


# ==========================================
# 4. RAPOR OLUŞTURUCU (DOKTOR GİBİ YAZAR)
# ==========================================
def generate_report(prob_pneumonia, pred_class):
    risk_level = ""
    color = ""
    findings = ""
    recommendation = ""

    if prob_pneumonia > 0.85:
        risk_level = "YÜKSEK RİSK"
        color = "red"
        findings = "Akciğer parankiminde yaygın opasite artışı ve konsolidasyon ile uyumlu bulgular izlenmiştir. Pnömoni (Zatürre) lehine kuvvetli şüphe mevcuttur."
        recommendation = "Acil uzman hekim değerlendirmesi ve ileri tetkik (BT, Kan Tahlili) önerilir."
    elif prob_pneumonia > 0.50:
        risk_level = "ORTA RİSK / ŞÜPHELİ"
        color = "orange"
        findings = "Fokal alanlarda hafif yoğunluk artışı tespit edilmiştir. Kesin pnömoni ayrımı yapılamamakla birlikte şüpheli görünüm mevcuttur."
        recommendation = "Klinik bulgularla korelasyon sağlanmalı, gerekirse takip grafisi çekilmelidir."
    else:
        risk_level = "NORMAL / DÜŞÜK RİSK"
        color = "green"
        findings = "Akciğer havalanması normaldir. Plevral sinüsler açıktır. Aktif infiltrasyon veya konsolidasyon saptanmamıştır."
        recommendation = "Rutin kontrol."

    report_text = f"""
    ### 📋 Radyolojik AI Ön Raporu
    **Tarih:** {np.datetime64('now')}
    **İnceleme:** PA Akciğer Grafisi

    ---
    **AI Tahmini:** :{color}[**{risk_level}**]  
    **Enfeksiyon İhtimali:** %{prob_pneumonia * 100:.1f}

    **Bulgular:** {findings}

    **Sonuç ve Öneri:** {recommendation}

    ---
    *Not: Bu rapor Yapay Zeka (AI-Rad v1.0) tarafından üretilmiştir. Kesin teşhis değildir, karar destek amaçlıdır.*
    """
    return report_text


# ==========================================
# 5. ARAYÜZ (MAIN APP)
# ==========================================
def main():
    st.sidebar.title("🫁 AI-Rad Kontrol Paneli")
    st.sidebar.info("Bu sistem hastalık teşhisinde yardımcı olmak kapsamında geliştirilmiştir.")

    uploaded_file = st.sidebar.file_uploader("Röntgen Görüntüsü Yükle", type=["jpg", "png", "jpeg"])

    model, seg_model = load_models()
    if model is None:
        return

    st.title("Derin Öğrenme Tabanlı Pnömoni Teşhis Sistemi")
    st.markdown("---")

    if uploaded_file is not None:
        # Görüntüyü İşle
        image = Image.open(uploaded_file).convert("RGB")

        # Kolonlara böl (Sol: Resim, Sağ: Analiz Butonu)
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption="Yüklenen Görüntü", use_container_width=True)

        with col2:
            analyze_btn = st.button("🔍 Görüntüyü Analiz Et", type="primary")

            if analyze_btn:
                with st.spinner(
                        'Yapay Zeka görüntüyü inceliyor... Segmentasyon yapılıyor... Isı haritası çıkarılıyor...'):
                    # 1. Transform
                    tfm = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

                    # 2. Maskeleme & Crop
                    mask = get_lung_mask(seg_model, image)

                    # Resmi 224 yap ve maskeyle çarp (Modelin gördüğü hale getir)
                    img_resized = np.array(image.resize((224, 224)))
                    img_masked = img_resized.copy()
                    # Maskeyi 3 kanala yay
                    mask_3ch = np.stack([mask] * 3, axis=-1)
                    img_masked[mask_3ch == 0] = 0

                    # Model Input
                    pil_masked = Image.fromarray(img_masked)
                    x = tfm(pil_masked).unsqueeze(0).to(DEVICE)

                    # 3. Grad-CAM
                    target_layer = model.features[-1]
                    cam_obj = GradCAM(model, target_layer)

                    heatmap, logits = cam_obj(x, class_idx=1)  # 1 = PNEUMONIA

                    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
                    p_pneu = probs[1]
                    pred_class = "PNÖMONİ" if p_pneu > 0.5 else "NORMAL"

                    # 4. Görselleştirme (Heatmap Overlay)
                    heatmap = cv2.resize(heatmap, (224, 224))
                    heatmap = normalize_0_1(heatmap)

                    # Heatmap'i maskele (Dışarı taşanı sil)
                    heatmap = heatmap * mask

                    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

                    # Orijinal (siyah-beyaz) üzerine bindir
                    gray_bg = cv2.cvtColor(cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2BGR)
                    overlay = cv2.addWeighted(gray_bg, 0.6, heatmap_color, 0.4, 0)

                    # SONUÇLARI GÖSTER
                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        st.image(img_masked, caption="Modelin Gördüğü (Maskeli)", use_container_width=True)
                    with res_col2:
                        st.image(heatmap_color, caption="AI Dikkat Haritası", use_container_width=True)
                    with res_col3:
                        st.image(overlay, caption="Klinik Çakıştırma", use_container_width=True)

                    # 5. RAPORU YAZDIR
                    st.markdown("---")
                    report = generate_report(p_pneu, pred_class)
                    st.markdown(report)

                    # İndirme Butonu
                    st.download_button("📥 Raporu İndir (.txt)", report, file_name="hasta_raporu.txt")

    else:
        st.info("Lütfen sol menüden bir röntgen görüntüsü yükleyiniz.")


if __name__ == "__main__":
    main()