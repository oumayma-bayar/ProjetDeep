import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

st.set_page_config(page_title="Penn Action — Analyse d'Exercice", page_icon="🏋️", layout="wide")

ACTIONS = [
    'baseball_pitch','baseball_swing','bench_press','bowling',
    'clean_and_jerk','golf_swing','jump_rope','jumping_jacks',
    'pull_ups','push_ups','sit_ups','squats',
    'strum_guitar','tennis_forehand','tennis_serve'
]
INJURY_RISK = {
    'baseball_pitch':  ('Shoulder / Elbow','High','🔴'),
    'baseball_swing':  ('Lower Back','Medium','🟡'),
    'bench_press':     ('Shoulder / Chest','Medium','🟡'),
    'bowling':         ('Wrist / Shoulder','Low','🟢'),
    'clean_and_jerk':  ('Lower Back / Knee','High','🔴'),
    'golf_swing':      ('Lower Back','Medium','🟡'),
    'jump_rope':       ('Ankle / Knee','Low','🟢'),
    'jumping_jacks':   ('Ankle','Low','🟢'),
    'pull_ups':        ('Shoulder / Elbow','Medium','🟡'),
    'push_ups':        ('Wrist / Shoulder','Low','🟢'),
    'sit_ups':         ('Lower Back / Neck','Medium','🟡'),
    'squats':          ('Knee / Lower Back','Medium','🟡'),
    'strum_guitar':    ('Wrist / Finger','Low','🟢'),
    'tennis_forehand': ('Elbow / Shoulder','High','🔴'),
    'tennis_serve':    ('Shoulder / Elbow','High','🔴'),
}
PENN_TO_INJURY = {
    'baseball_pitch':'HIIT','baseball_swing':'HIIT','bench_press':'Weight Training',
    'bowling':'HIIT','clean_and_jerk':'Weight Training','golf_swing':'HIIT',
    'jump_rope':'Running','jumping_jacks':'HIIT','pull_ups':'Weight Training',
    'push_ups':'Weight Training','sit_ups':'Weight Training','squats':'Weight Training',
    'strum_guitar':'Yoga','tennis_forehand':'HIIT','tennis_serve':'HIIT',
}
EMOJIS = ['⚾','⚾','🏋️','🎳','🏋️','⛳','🪢','🤸','💪','💪','🧘','🏋️','🎸','🎾','🎾']
IMG_SIZE = (224, 224)

# ── Transform IDENTIQUE à l'entraînement ─────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── Reconstruction de l'architecture ─────────────────────────────
def build_model(arch, num_classes):
    if arch in ('resnet50','baseline','p4_bal','p3_adamw','p1_aug','p2_arch'):
        m = models.resnet50(weights=None)
        f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(f, num_classes))
    elif arch == 'p5_ft':
        m = models.resnet50(weights=None)
        f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Linear(f,512), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    elif arch == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=None)
        f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(f, num_classes))
    elif arch == 'efficientnet_b2':
        m = models.efficientnet_b2(weights=None)
        f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(f, num_classes))
    elif arch == 'densenet121':
        m = models.densenet121(weights=None)
        f = m.classifier.in_features
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(f, num_classes))
    else:
        m = models.resnet50(weights=None)
        f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(f, num_classes))
    return m

@st.cache_resource
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(model_path):
        m = build_model('resnet50', 15)
        m.eval().to(device)
        return m, device, "⚠️ Modèle non trouvé — poids aléatoires", 'resnet50', 0.0

    ckpt        = torch.load(model_path, map_location=device, weights_only=False)
    arch        = ckpt.get('arch', 'resnet50')
    num_classes = ckpt.get('num_classes', 15)
    val_acc     = ckpt.get('val_acc', 0.0)

    model = build_model(arch, num_classes)

    # Gérer DataParallel (clés 'module.xxx')
    state = ckpt['model_state_dict']
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.',''):v for k,v in state.items()}

    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model, device, f"✅ arch={arch} | Val Acc={val_acc:.2%}", arch, val_acc

def predict(pil_img, model, device):
    t = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return idx, float(probs[idx]), probs

def gradcam(model, device, pil_img, pred_idx, arch):
    t = transform(pil_img).unsqueeze(0).to(device)
    acts, grads = {}, {}
    # Couche cible selon architecture
    if 'efficientnet' in arch:
        layer = model.features[-1]
    elif 'densenet' in arch:
        layer = model.features.denseblock4
    else:
        layer = model.layer4[-1]

    h1 = layer.register_forward_hook(lambda m,i,o: acts.update({'f':o}))
    h2 = layer.register_full_backward_hook(lambda m,gi,go: grads.update({'f':go[0]}))
    model.eval()
    out = model(t)
    model.zero_grad()
    out[0, pred_idx].backward()
    h1.remove(); h2.remove()

    w   = grads['f'].mean(dim=[2,3], keepdim=True)
    hm  = torch.relu((w * acts['f']).sum(dim=1).squeeze()).detach().cpu().numpy()
    hm  = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    img = np.array(pil_img.resize(IMG_SIZE)).astype(np.float32) / 255.0
    hm2 = cv2.resize(hm, IMG_SIZE)
    col = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*hm2), cv2.COLORMAP_JET),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(0.55*img + 0.45*col, 0, 1), hm2

# ─── UI ──────────────────────────────────────────────────────────
st.title("🏋️ Penn Action — Analyse d'Exercice Sportif")
st.markdown("Importez une **image** d'un sportif → détection de l'exercice + risque de blessure + Grad-CAM.")

with st.sidebar:
    st.header("⚙️ Configuration")
    model_path   = st.text_input("Fichier modèle (.pth)", value="penn_action_best.pth")
    show_gradcam = st.checkbox("Afficher Grad-CAM", value=True)
    show_topk    = st.checkbox("Afficher Top-5", value=True)
    st.divider()
    st.markdown("**15 classes :**")
    for e, a in zip(EMOJIS, ACTIONS):
        st.markdown(f"{e} {a.replace('_',' ')}")

model, device, status, arch, val_acc = load_model(model_path)
st.info(status)

uploaded = st.file_uploader("📤 Importer une image (jpg, png, webp)", type=["jpg","jpeg","png","webp"])

if uploaded:
    pil_img = Image.open(uploaded).convert('RGB')
    pred_idx, conf, probs = predict(pil_img, model, device)
    action = ACTIONS[pred_idx]
    zone, risk, emoji = INJURY_RISK[action]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🖼️ Image importée")
        st.image(pil_img, use_container_width=True)
    with c2:
        st.subheader("🎯 Résultat")
        bg = '#fee2e2' if risk=='High' else '#fef9c3' if risk=='Medium' else '#dcfce7'
        st.markdown(f"""<div style="background:{bg};border-radius:12px;padding:20px;margin-bottom:12px;">
            <h2 style="margin:0;font-size:26px;">{action.replace('_',' ').title()}</h2>
            <p style="margin:4px 0;color:#555;">Confiance : <strong>{conf:.1%}</strong></p>
        </div>""", unsafe_allow_html=True)
        st.progress(conf, text=f"Confiance : {conf:.1%}")
        st.divider()
        st.markdown("**🩺 Risque de blessure**")
        r1,r2,r3 = st.columns(3)
        r1.metric("Zone", zone)
        r2.metric("Niveau", f"{emoji} {risk}")
        r3.metric("Confiance", f"{conf:.1%}")

    if show_topk:
        st.divider()
        st.subheader("📊 Top-5 prédictions")
        top5 = np.argsort(probs)[::-1][:5]
        fig, ax = plt.subplots(figsize=(10,3))
        cols_bar = ['#3b82f6' if i==0 else '#94a3b8' for i in range(5)]
        acts_r = [ACTIONS[i].replace('_',' ') for i in top5][::-1]
        prob_r = [float(probs[i]) for i in top5][::-1]
        bars = ax.barh(acts_r, prob_r, color=cols_bar[::-1], edgecolor='black', linewidth=0.5)
        ax.set_xlim(0,1); ax.set_xlabel('Probabilité')
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.4)
        for bar, v in zip(bars, prob_r):
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2, f'{v:.1%}', va='center', fontsize=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    if show_gradcam:
        st.divider()
        st.subheader("🔍 Grad-CAM")
        st.caption("Zones rouges/jaunes = ce que le modèle regarde pour décider.")
        try:
            overlay, hm = gradcam(model, device, pil_img, pred_idx, arch)
            g1,g2,g3 = st.columns(3)
            g1.image(np.array(pil_img.resize(IMG_SIZE)), caption="Originale", use_container_width=True)
            g2.image((hm*255).astype(np.uint8), caption="Heatmap", use_container_width=True)
            g3.image((overlay*255).astype(np.uint8), caption="Overlay", use_container_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM indisponible : {e}")

    st.divider()
    st.subheader("🔗 Connexion projet Injury")
    ex_type = PENN_TO_INJURY.get(action, 'HIIT')
    i1,i2 = st.columns(2)
    i1.metric("Exercise Type (ANN)", ex_type)
    i2.metric("Action CNN", action.replace('_',' '))
    st.code(f"profile = {{'exercise_type': '{ex_type}', 'age': 28, 'BMI': 24.5, ...}}", language='python')

else:
    st.markdown("""<div style="text-align:center;padding:60px;color:#888;">
        <h2>📸 Importez une image pour commencer</h2>
        <p>Formats : JPG, PNG, WEBP</p></div>""", unsafe_allow_html=True)
    st.divider()
    st.subheader("🎯 15 classes reconnues")
    cols = st.columns(5)
    for i,(a,e) in enumerate(zip(ACTIONS, EMOJIS)):
        cols[i%5].markdown(f"""<div style="text-align:center;padding:10px;background:#f8fafc;
            border-radius:8px;margin:4px;border:1px solid #e2e8f0;">
            <div style="font-size:24px;">{e}</div>
            <div style="font-size:11px;color:#475569;">{a.replace('_',' ')}</div>
        </div>""", unsafe_allow_html=True)
