import os
import io
import re
import pickle
import urllib.request
from pathlib import Path
from PIL import Image
from gtts import gTTS
import streamlit as st

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# ==========================================
# 0. DIRECTORY SETUP
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

ENCODER_URL = "https://huggingface.co/your-username/sightvoice/resolve/main/encoder.pth"
DECODER_URL = "https://huggingface.co/your-username/sightvoice/resolve/main/decoder.pth"

def download_weight_if_missing(file_path, url):
    """Downloads model weights if not locally present."""
    if not file_path.exists():
        st.info(f"Downloading {file_path.name}...")
        try:
            urllib.request.urlretrieve(url, file_path)
        except Exception as e:
            st.error(f"Failed to download {file_path.name}: {e}")

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="SightVoice — Assistive Scene Reader",
    page_icon="https://img.icons8.com/fluency/48/visible.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. LAVISH, ACCESSIBLE HIGH-CONTRAST CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    :root {
        --bg-0: #070b14;
        --bg-1: #0d1424;
        --bg-2: #131c30;
        --stroke: rgba(120, 160, 255, 0.18);
        --stroke-strong: rgba(120, 160, 255, 0.4);
        --accent: #6ea8ff;
        --accent-2: #8affc1;
        --accent-3: #b18cff;
        --text: #eaf0ff;
        --muted: #8fa1c7;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--bg-0) !important;
        color: var(--text);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: -20%;
        background:
            radial-gradient(40% 40% at 15% 10%, rgba(110,168,255,0.18), transparent 60%),
            radial-gradient(35% 35% at 85% 20%, rgba(177,140,255,0.16), transparent 60%),
            radial-gradient(45% 45% at 50% 100%, rgba(138,255,193,0.10), transparent 60%);
        filter: blur(40px);
        z-index: 0;
        pointer-events: none;
        animation: drift 24s ease-in-out infinite alternate;
    }
    @keyframes drift {
        0%   { transform: translate3d(0,0,0) scale(1); }
        100% { transform: translate3d(2%, -2%, 0) scale(1.06); }
    }

    .block-container {
        position: relative;
        z-index: 1;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
        max-width: 880px;
    }

    .hero {
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 26px 28px;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(23,32,54,0.85), rgba(13,20,36,0.85));
        border: 1px solid var(--stroke);
        box-shadow:
            0 30px 60px -30px rgba(0,0,0,0.9),
            inset 0 1px 0 rgba(255,255,255,0.04);
        backdrop-filter: blur(18px);
        margin-bottom: 26px;
    }
    .hero-icon {
        flex: 0 0 auto;
        width: 62px; height: 62px;
        border-radius: 18px;
        display: grid; place-items: center;
        background: linear-gradient(135deg, rgba(110,168,255,0.25), rgba(177,140,255,0.25));
        border: 1px solid var(--stroke-strong);
        box-shadow: 0 0 30px rgba(110,168,255,0.35);
    }
    .hero-text h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(90deg, #eaf0ff 0%, #6ea8ff 55%, #b18cff 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-text p {
        margin: 6px 0 0 0;
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(19,28,48,0.7);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid var(--stroke);
        backdrop-filter: blur(12px);
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        padding: 0 20px;
        border-radius: 11px;
        color: var(--muted);
        font-weight: 600;
        font-size: 0.92rem;
        transition: all 0.25s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(110,168,255,0.22), rgba(177,140,255,0.22)) !important;
        color: #fff !important;
        box-shadow: inset 0 0 0 1px var(--stroke-strong), 0 8px 20px -12px rgba(110,168,255,0.9);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    section[data-testid="stFileUploaderDropzone"],
    div[data-testid="stCameraInput"] video,
    div[data-testid="stCameraInput"] img {
        border-radius: 16px !important;
    }
    section[data-testid="stFileUploaderDropzone"] {
        background: rgba(19,28,48,0.6) !important;
        border: 1.5px dashed var(--stroke-strong) !important;
        padding: 26px !important;
    }

    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 14px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.01em;
        color: #06101f;
        background: linear-gradient(135deg, #8affc1 0%, #6ea8ff 100%);
        border: none;
        box-shadow: 0 14px 30px -14px rgba(110,168,255,0.9);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px -16px rgba(110,168,255,1);
    }
    .stButton > button:active { transform: translateY(0); }

    .caption-card {
        position: relative;
        margin: 26px 0 18px 0;
        padding: 26px 28px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(19,28,48,0.92), rgba(11,17,31,0.92));
        border: 1px solid var(--stroke-strong);
        box-shadow:
            0 30px 60px -30px rgba(0,0,0,0.95),
            inset 0 1px 0 rgba(255,255,255,0.05);
        overflow: hidden;
    }
    .caption-card::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, transparent 30%, rgba(110,168,255,0.08) 50%, transparent 70%);
        animation: sheen 6s linear infinite;
        pointer-events: none;
    }
    @keyframes sheen {
        0%   { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    .caption-label {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent);
        font-weight: 700;
        margin-bottom: 12px;
    }
    .caption-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem;
        line-height: 1.4;
        font-weight: 600;
        color: #eaf0ff;
        margin: 0;
    }
    .caption-text::before, .caption-text::after { content: '"'; color: var(--accent-2); }

    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.78rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 700;
        margin: 32px 0 14px 2px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(138,255,193,0.08);
        border: 1px solid rgba(138,255,193,0.35);
        color: var(--accent-2);
    }
    .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--accent-2);
        box-shadow: 0 0 12px var(--accent-2);
        animation: pulse 1.6s ease-in-out infinite;
    }
    @keyframes pulse {
        0%,100% { opacity: 1; }
        50% { opacity: 0.35; }
    }

    audio { width: 100%; border-radius: 12px; }

    div[data-testid="stImage"] img {
        border-radius: 16px;
        border: 1px solid var(--stroke);
        box-shadow: 0 20px 40px -22px rgba(0,0,0,0.9);
    }

    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SVG ICON HELPERS
# ==========================================
def svg(path_d, size=22, color="#eaf0ff", stroke=1.9):
    return f'''
    <svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
         viewBox="0 0 24 24" fill="none" stroke="{color}"
         stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">
        {path_d}
    </svg>
    '''

ICON_EYE = '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/>'
ICON_CAMERA = '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>'
ICON_UPLOAD = '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>'
ICON_SPARK = '<path d="M12 2l1.9 5.6L19.5 9.5 13.9 11.4 12 17l-1.9-5.6L4.5 9.5l5.6-1.9z"/>'
ICON_VOLUME = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M18.5 5.5a9 9 0 0 1 0 13"/>'
ICON_IMAGE = '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>'

# ==========================================
# 4. MODEL & VOCABULARY CLASSES
# ==========================================
class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.split()

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(2048, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(3))
        features = self.embed(features)
        return features

class BahdanauAttention(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(embed_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, features, hidden):
        hidden_with_time_axis = hidden.unsqueeze(1)
        score = torch.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = torch.softmax(self.V(score), dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_weights

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = BahdanauAttention(embed_size, hidden_size)
        self.gru = nn.GRU(embed_size + embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, hidden, word):
        word_embed = self.embed(word)
        context_vector, attention_weights = self.attention(features, hidden)
        gru_input = torch.cat((word_embed, context_vector), dim=1).unsqueeze(1)
        output, hidden = self.gru(gru_input, hidden.unsqueeze(0))
        output = self.fc(output.squeeze(1))
        return output, hidden.squeeze(0), attention_weights

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# ==========================================
# 5. LOAD CACHED MODEL WEIGHTS & VOCAB
# ==========================================
@st.cache_resource
def load_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder_path = BASE_DIR / "encoder.pth"
    decoder_path = BASE_DIR / "decoder.pth"
    vocab_path = BASE_DIR / "vocab.pkl"

    download_weight_if_missing(encoder_path, ENCODER_URL)
    download_weight_if_missing(decoder_path, DECODER_URL)

    if not encoder_path.exists() or not decoder_path.exists():
        st.error("Model weight files could not be found or downloaded.")
        st.stop()

    if not vocab_path.exists():
        st.error("Vocabulary file (vocab.pkl) is missing from the directory.")
        st.stop()

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab)

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()

    return encoder, decoder, vocab, device

encoder, decoder, vocab, device = load_pipeline()

def generate_caption(image, max_len=20):
    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    caption = []

    with torch.no_grad():
        features = encoder(image_tensor)
        hidden = torch.zeros(1, 512).to(device)
        word = torch.tensor([vocab.stoi["<SOS>"]]).to(device)

        for _ in range(max_len):
            output, hidden, _ = decoder(features, hidden, word)
            predicted = output.argmax(1)
            token = vocab.itos[predicted.item()]

            if token == "<EOS>":
                break

            caption.append(token)
            word = predicted

    return " ".join(caption)

def text_to_speech_bytes(text):
    tts = gTTS(text=text, lang='en')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

# ==========================================
# 6. UI — HERO
# ==========================================
st.markdown(f"""
<div class="hero">
    <div class="hero-icon">{svg(ICON_EYE, size=30, color="#8affc1", stroke=1.8)}</div>
    <div class="hero-text">
        <h1>SightVoice</h1>
        <p>An assistive scene reader that sees, understands, and speaks — built for accessibility and elegance.</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
    <span class="status-pill"><span class="dot"></span> Model online &nbsp;·&nbsp; {str(device).upper()}</span>
    <span style="color: var(--muted); font-size: 0.78rem; letter-spacing:0.08em;">ATTENTION-BASED CAPTIONING</span>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 7. TABS — INPUT
# ==========================================
tab1, tab2 = st.tabs([
    "  Live Camera  ",
    "  Upload Image  "
])

image_input = None

with tab1:
    st.markdown(
        f'<div class="section-title">{svg(ICON_CAMERA, size=16, color="#6ea8ff")} Capture from device</div>',
        unsafe_allow_html=True
    )
    camera_file = st.camera_input("Take a photo", label_visibility="collapsed")
    if camera_file:
        image_input = Image.open(camera_file)

with tab2:
    st.markdown(
        f'<div class="section-title">{svg(ICON_UPLOAD, size=16, color="#6ea8ff")} Upload from storage</div>',
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        image_input = Image.open(uploaded_file)

# ==========================================
# 8. OUTPUT — IMAGE, CAPTION, AUDIO
# ==========================================
if image_input:
    st.markdown(
        f'<div class="section-title">{svg(ICON_IMAGE, size=16, color="#6ea8ff")} Visual Scene</div>',
        unsafe_allow_html=True
    )
    st.image(image_input, use_container_width=True)

    with st.spinner("Analyzing scene and synthesizing narration..."):
        caption_text = generate_caption(image_input)
        
        if "audio_bytes" not in st.session_state or st.session_state.get("last_caption") != caption_text:
            st.session_state.audio_bytes = text_to_speech_bytes(caption_text)
            st.session_state.last_caption = caption_text

    st.markdown(f"""
    <div class="caption-card">
        <div class="caption-label">
            {svg(ICON_SPARK, size=14, color="#6ea8ff")}
            <span>Generated Caption</span>
        </div>
        <p class="caption-text">{caption_text}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="section-title">{svg(ICON_VOLUME, size=16, color="#8affc1")} Voice Narration</div>',
        unsafe_allow_html=True
    )
    st.audio(st.session_state.audio_bytes, format="audio/mp3", autoplay=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    if st.button("Replay Speech Description", use_container_width=True):
        st.audio(st.session_state.audio_bytes, format="audio/mp3", autoplay=True)