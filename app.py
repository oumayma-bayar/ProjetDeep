import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os

# ──────────────────────────────────────────────
# Config page
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Penn Action — Analyse d'Exercice",
    page_icon="🏋️",
    layout="wide",
)

# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────
ACTIONS = [
    'baseball_pitch', 'baseball_swing', 'bench_press', 'bowling',
    'clean_and_jerk', 'golf_swing',     'jump_rope',   'jumping_jacks',
    'pull_ups',       'push_ups',       'sit_ups',     'squats',
    'strum_guitar',   'tennis_forehand','tennis_serve'
]

# Risque de blessure associé à chaque exercice (pour le projet injury)
INJURY_RISK = {
    'baseball_pitch':  ('Shoulder / Elbow', 'High',   '🔴'),
    'baseball_swing':  ('Lower Back',        'Medium', '🟡'),
    'bench_press':     ('Shoulder / Chest',  'Medium', '🟡'),
    'bowling':         ('Wrist / Shoulder',  'Low',    '🟢'),
    'clean_and_jerk':  ('Lower Back / Knee', 'High',   '🔴'),
    'golf_swing':      ('Lower Back',        'Medium', '🟡'),
    'jump_rope':       ('Ankle / Knee',      'Low',    '🟢'),
    'jumping_jacks':   ('Ankle',             'Low',    '🟢'),
    'pull_ups':        ('Shoulder / Elbow',  'Medium', '🟡'),
    'push_ups':        ('Wrist / Shoulder',  'Low',    '🟢'),
    'sit_ups':         ('Lower Back / Neck', 'Medium', '🟡'),
    'squats':          ('Knee / Lower Back', 'Medium', '🟡'),
    'strum_guitar':    ('Wrist / Finger',    'Low',    '🟢'),
    'tennis_forehand': ('Elbow / Shoulder',  'High',   '🔴'),
    'tennis_serve':    ('Shoulder / Elbow',  'High',   '🔴'),
}

IMG_SIZE = (224, 224)

# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ──────────────────────────────────────────────
# Chargement du modèle
# ──────────────────────────────────────────────
@st.cache_resource
def load_model(model_path='penn_action_best.pth'):
    """Charge le modèle EfficientNet-B0 entraîné."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        arch       = checkpoint.get('arch', 'efficientnet_b0')
        num_classes= checkpoint.get('num_classes', 15)

        if arch == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=None)
            f = model.classifier[1].in_features
            model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(f, num_classes))
        elif arch == 'resnet50':
            model = models.resnet50(weights=None)
            f = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(f, num_classes))
        else:
            model = models.efficientnet_b0(weights=None)
            f = model.classifier[1].in_features
            model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(f, num_classes))

        model.load_state_dict(checkpoint['model_state_dict'])
        val_acc = checkpoint.get('val_acc', None)
        status  = f"✅ Modèle chargé ({arch}) — Val Acc: {val_acc:.2%}" if val_acc else f"✅ Modèle chargé ({arch})"
    else:
        # Démo : modèle aléatoire EfficientNet
        model   = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(f, 15))
        status  = "⚠️ Aucun modèle trouvé — démo avec poids aléatoires"

    model.eval()
    model.to(device)
    return model, device, status


@st.cache_data
def predict(img_array, _model, _device):
    """Prédit la classe et les probabilités pour une image numpy."""
    pil_img = Image.fromarray(img_array).convert('RGB')
    tensor  = transform(pil_img).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return pred_idx, confidence, probs


def compute_gradcam(model, device, img_array, pred_idx):
    """Grad-CAM sur le dernier bloc conv d'EfficientNet."""
    pil_img = Image.fromarray(img_array).convert('RGB')
    tensor  = transform(pil_img).unsqueeze(0).to(device)

    activations, gradients = {}, {}

    # Hook sur le dernier bloc de features
    target_layer = model.features[-1]

    def fwd_hook(m, i, o): activations['feat'] = o
    def bwd_hook(m, gi, go): gradients['feat'] = go[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.eval()
    out = model(tensor)
    model.zero_grad()
    out[0, pred_idx].backward()

    h1.remove(); h2.remove()

    weights  = gradients['feat'].mean(dim=[2, 3], keepdim=True)
    heatmap  = (weights * activations['feat']).sum(dim=1).squeeze()
    heatmap  = torch.relu(heatmap).detach().cpu().numpy()
    heatmap  = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    h, w = img_array.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h))
    hm_colored = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
    hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_norm   = img_array.astype(np.float32) / 255.0
    overlay    = np.clip(0.55 * img_norm + 0.45 * hm_colored, 0, 1)
    return overlay, hm_resized


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
st.title("🏋️ Penn Action — Analyse d'Exercice Sportif")
st.markdown("Importez une image d'un sportif et obtenez le **type d'exercice** détecté, "
            "la **confiance du modèle**, la **carte Grad-CAM** et l'**estimation du risque de blessure**.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    model_path = st.text_input("Chemin du modèle .pth", value="penn_action_best.pth")
    show_gradcam = st.checkbox("Afficher Grad-CAM", value=True)
    show_topk    = st.checkbox("Afficher Top-5 classes", value=True)
    st.divider()
    st.markdown("**15 classes disponibles :**")
    for i, a in enumerate(ACTIONS):
        st.markdown(f"`{i:2d}` {a.replace('_', ' ')}")

# Chargement modèle
model, device, model_status = load_model(model_path)
st.info(model_status)

# Upload
uploaded = st.file_uploader(
    "📤 Importer une image (jpg, png, webp)",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is not None:
    # Lire l'image
    pil_img   = Image.open(uploaded).convert('RGB')
    img_array = np.array(pil_img)

    # Prédiction
    pred_idx, confidence, probs = predict(img_array, model, device)
    pred_action = ACTIONS[pred_idx]
    injury_zone, risk_level, risk_emoji = INJURY_RISK[pred_action]

    # ── Layout principal ──────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼️ Image importée")
        st.image(pil_img, use_container_width=True)

    with col2:
        st.subheader("🎯 Résultat de l'analyse")

        # Résultat principal
        st.markdown(f"""
        <div style="background:{'#fee2e2' if risk_level=='High' else '#fef9c3' if risk_level=='Medium' else '#dcfce7'};
                    border-radius:12px; padding:20px; margin-bottom:16px;">
            <h2 style="margin:0; font-size:28px;">
                {pred_action.replace('_', ' ').title()}
            </h2>
            <p style="margin:4px 0; font-size:16px; color:#555;">
                Confiance : <strong>{confidence:.1%}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Barre de confiance
        st.progress(float(confidence), text=f"Confiance : {confidence:.1%}")

        st.divider()

        # Risque de blessure
        st.markdown("**🩺 Risque de blessure associé**")
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Zone", injury_zone)
        col_r2.metric("Niveau", f"{risk_emoji} {risk_level}")
        col_r3.metric("Confiance CNN", f"{confidence:.1%}")

    # ── Top-K classes ─────────────────────────
    if show_topk:
        st.divider()
        st.subheader("📊 Top-5 prédictions")
        top5_idx  = np.argsort(probs)[::-1][:5]
        top5_acts = [ACTIONS[i].replace('_', ' ') for i in top5_idx]
        top5_probs= [float(probs[i]) for i in top5_idx]

        fig, ax = plt.subplots(figsize=(10, 3))
        colors_bar = ['#3b82f6' if i == 0 else '#94a3b8' for i in range(5)]
        bars = ax.barh(top5_acts[::-1], top5_probs[::-1],
                       color=colors_bar[::-1], edgecolor='black', linewidth=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probabilité')
        ax.set_title('Top-5 classes prédites')
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.4, label='50%')
        for bar, v in zip(bars, top5_probs[::-1]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{v:.1%}', va='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Grad-CAM ──────────────────────────────
    if show_gradcam:
        st.divider()
        st.subheader("🔍 Grad-CAM — Zones d'attention du modèle")
        st.caption("Les zones en rouge/jaune sont celles que le modèle utilise pour sa décision.")

        try:
            # Grad-CAM uniquement sur EfficientNet
            if hasattr(model, 'features'):
                overlay, heatmap = compute_gradcam(model, device, img_array, pred_idx)
                col_g1, col_g2, col_g3 = st.columns(3)
                col_g1.image(img_array, caption="Originale", use_container_width=True)
                col_g2.image((heatmap * 255).astype(np.uint8),
                             caption="Heatmap", use_container_width=True,
                             clamp=True)
                col_g3.image((overlay * 255).astype(np.uint8),
                             caption="Overlay", use_container_width=True,
                             clamp=True)
            else:
                st.info("Grad-CAM disponible uniquement pour EfficientNet. Ré-entraîner avec arch='efficientnet_b0'.")
        except Exception as e:
            st.warning(f"Grad-CAM non disponible : {e}")

    # ── Connexion projet blessures ─────────────
    st.divider()
    st.subheader("🔗 Connexion avec le projet Injury")
    st.markdown("""
    Pour brancher ce résultat sur ton ANN `fitness_injury_v2_complete.ipynb` :
    ```python
    # Résultat CNN → features tabulaires
    profile = {
        'exercise_type': predicted_exercise,   # ← vient du CNN
        'age': 28,
        'BMI': 24.5,
        'training_intensity': 'High',
        'previous_injury': 'No',
        'duration_min': 45
    }
    # injury_risk = ton_ann_model.predict(profile)
    ```
    """)

    PENN_TO_INJURY_TYPE = {
        'baseball_pitch': 'HIIT', 'baseball_swing': 'HIIT',
        'bench_press': 'Weight Training', 'bowling': 'HIIT',
        'clean_and_jerk': 'Weight Training', 'golf_swing': 'HIIT',
        'jump_rope': 'Running', 'jumping_jacks': 'HIIT',
        'pull_ups': 'Weight Training', 'push_ups': 'Weight Training',
        'sit_ups': 'Weight Training', 'squats': 'Weight Training',
        'strum_guitar': 'Yoga', 'tennis_forehand': 'HIIT', 'tennis_serve': 'HIIT',
    }
    exercise_type = PENN_TO_INJURY_TYPE.get(pred_action, 'HIIT')

    col_i1, col_i2 = st.columns(2)
    col_i1.metric("Exercise Type (ANN input)", exercise_type)
    col_i2.metric("Action détectée (CNN output)", pred_action.replace('_', ' '))

else:
    # Page d'accueil vide
    st.markdown("""
    <div style="text-align:center; padding:60px; color:#888;">
        <h2>📸 Importez une image pour commencer</h2>
        <p>Formats acceptés : JPG, PNG, WEBP</p>
        <p>Exemples d'images : squats, push-ups, tennis, etc.</p>
    </div>
    """, unsafe_allow_html=True)

    # Exemples visuels des classes
    st.divider()
    st.subheader("🎯 Classes reconnues par le modèle")
    cols = st.columns(5)
    emojis = ['⚾','⚾','🏋️','🎳','🏋️','⛳','🪢','🤸','💪','💪','🧘','🏋️','🎸','🎾','🎾']
    for i, (col, action, emoji) in enumerate(zip(cols * 3, ACTIONS, emojis)):
        col.markdown(f"""
        <div style="text-align:center; padding:10px; background:#f8fafc;
                    border-radius:8px; margin:4px; border:1px solid #e2e8f0;">
            <div style="font-size:24px;">{emoji}</div>
            <div style="font-size:11px; color:#475569;">{action.replace('_', ' ')}</div>
        </div>
        """, unsafe_allow_html=True)
