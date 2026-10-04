import hashlib
import html
import inspect
import io
import os
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from gtts import gTTS
from PIL import Image, ImageOps

# Model code lives in model.py and is shared with train.py, so training and
# inference can never disagree again.
from model import DecoderRNN, EncoderCNN, Vocabulary, beam_search, load_vocab_pickle

# ==========================================
# 0. FILES & SETTINGS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

V2_CHECKPOINT = BASE_DIR / "sightvoice_v2.pt"   # new model, produced by train.py
LEGACY_ENCODER = BASE_DIR / "encoder.pth"       # your original model files
LEGACY_DECODER = BASE_DIR / "decoder.pth"
LEGACY_VOCAB = BASE_DIR / "vocab.pkl"

# Optional download locations, used only if the files are not next to app.py.
# Set them as environment variables (or Streamlit Cloud secrets). The old code
# pointed at a placeholder "your-username" URL that could never work.
CHECKPOINT_URL = os.environ.get("SIGHTVOICE_CHECKPOINT_URL", "")
ENCODER_URL = os.environ.get("SIGHTVOICE_ENCODER_URL", "")
DECODER_URL = os.environ.get("SIGHTVOICE_DECODER_URL", "")
VOCAB_URL = os.environ.get("SIGHTVOICE_VOCAB_URL", "")

# Newer Streamlit replaced use_container_width=True with width="stretch"; support both.
STRETCH = {"width": "stretch"} if "width" in inspect.signature(st.button).parameters else {"use_container_width": True}


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


st.markdown("""
<style>
    .status-pill.pill-warn { background: rgba(255,200,87,0.08); border-color: rgba(255,200,87,0.45); color: #ffc857; }
    .status-pill.pill-bad  { background: rgba(255,107,107,0.08); border-color: rgba(255,107,107,0.5); color: #ff8a8a; }
    .fine-print { color: var(--muted); font-size: 0.78rem; line-height: 1.5; margin-top: 28px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. LOAD MODEL (cached)
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def _download(path, url):
    if path.exists():
        return True
    if not url:
        return False
    try:
        urllib.request.urlretrieve(url, path)
        return path.exists()
    except Exception:
        if path.exists():
            path.unlink()
        return False


@st.cache_resource(show_spinner="Loading SightVoice model...")
def load_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _download(V2_CHECKPOINT, CHECKPOINT_URL)
    if V2_CHECKPOINT.exists():
        # New model trained with train.py: one small file, vocab inside.
        ck = torch.load(V2_CHECKPOINT, map_location=device, weights_only=True)
        cfg = ck["config"]
        vocab = Vocabulary.from_list(ck["vocab_itos"])
        encoder = EncoderCNN(cfg["embed_size"], pretrained=True).to(device)
        encoder.embed.load_state_dict(ck["embed"])
        decoder = DecoderRNN(cfg["embed_size"], cfg["hidden_size"], cfg["vocab_size"]).to(device)
        decoder.load_state_dict(ck["decoder"])
        static, tag = cfg.get("attention", "dynamic") == "static", "v2"
    else:
        # Your original encoder.pth / decoder.pth / vocab.pkl.
        for path, url in ((LEGACY_ENCODER, ENCODER_URL), (LEGACY_DECODER, DECODER_URL), (LEGACY_VOCAB, VOCAB_URL)):
            _download(path, url)
        missing = [p.name for p in (LEGACY_ENCODER, LEGACY_DECODER, LEGACY_VOCAB) if not p.exists()]
        if missing:
            return {"error": "Missing model files: " + ", ".join(missing) +
                             ". Put them next to app.py (or set the SIGHTVOICE_*_URL environment variables)."}
        vocab = load_vocab_pickle(LEGACY_VOCAB)
        encoder = EncoderCNN(256, pretrained=False).to(device)
        encoder.load_state_dict(torch.load(LEGACY_ENCODER, map_location=device, weights_only=True))
        decoder = DecoderRNN(256, 512, len(vocab)).to(device)
        decoder.load_state_dict(torch.load(LEGACY_DECODER, map_location=device, weights_only=True))
        # The original notebook trained with a single, fixed attention context, so it
        # must be decoded the same way (the old app.py re-attended at every step and
        # also fed the GRU its inputs in the wrong order).
        static, tag = True, "v1"

    encoder.eval()
    decoder.eval()
    return dict(encoder=encoder, decoder=decoder, vocab=vocab, device=device, static=static, tag=tag)


pipe = load_pipeline()
if "error" in pipe:
    st.error(pipe["error"])
    st.stop()
encoder, decoder, vocab = pipe["encoder"], pipe["decoder"], pipe["vocab"]
device, STATIC_CONTEXT, MODEL_TAG = pipe["device"], pipe["static"], pipe["tag"]

# ==========================================
# 5. HELPERS: IMAGE, CAPTION, SPEECH
# ==========================================
def load_image(image_bytes):
    """Phone cameras store rotation in EXIF metadata - apply it, or the model sees
    sideways pictures (a common reason for bad results on real photos)."""
    img = Image.open(io.BytesIO(image_bytes))
    return ImageOps.exif_transpose(img).convert("RGB")


@st.cache_data(show_spinner=False, max_entries=24)
def run_captioning(image_bytes, beam_size, max_len):
    image = load_image(image_bytes)
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = encoder(x)
        return beam_search(decoder, feats, vocab, beam_size=beam_size,
                           max_len=max_len, static_context=STATIC_CONTEXT)


def pretty(text):
    text = text.strip()
    return text[0].upper() + text[1:] + "." if text else ""


def confidence_level(c):
    # Rough, uncalibrated thresholds on the model's own average token probability.
    # Tune them on a handful of your own test photos.
    if c >= 0.45:
        return "high", ""
    if c >= 0.25:
        return "medium", "pill-warn"
    return "low", "pill-bad"


def attention_overlay(image, weights, size=224):
    """Blend a 7x7 attention map over the image (v2 models only)."""
    side = int(round(len(weights) ** 0.5))
    w = weights.reshape(side, side)
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)
    heat = np.asarray(Image.fromarray((w * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)) / 255.0
    base = np.asarray(image.resize((size, size))).astype(float)
    color = np.zeros_like(base)
    color[..., 0], color[..., 1] = 255, 190
    a = (0.65 * heat)[..., None]
    return Image.fromarray((base * (1 - a) + color * a).astype(np.uint8))


LANGS = {"English": "en", "हिन्दी (Hindi)": "hi", "मराठी (Marathi)": "mr"}


@st.cache_data(show_spinner=False, max_entries=32)
def _translate(text, lang_code):          # raises on failure -> failures are not cached
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="en", target=lang_code).translate(text)


@st.cache_data(show_spinner=False, max_entries=32)
def _tts(text, lang_code, slow):          # raises on failure -> failures are not cached
    fp = io.BytesIO()
    gTTS(text=text, lang=lang_code, slow=slow).write_to_fp(fp)
    return fp.getvalue()


def narrate(text, lang_code, slow):
    """Returns (mp3 bytes or None, spoken text, optional note for the user)."""
    note, spoken = None, text
    if lang_code != "en":
        try:
            spoken = _translate(text, lang_code)
        except Exception:
            lang_code, note = "en", "Translation is unavailable right now, so the description is spoken in English."
    try:
        return _tts(spoken, lang_code, slow), spoken, note
    except Exception:
        return None, spoken, "Speech could not be generated (gTTS needs an internet connection)."


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
    <span class="status-pill"><span class="dot"></span> Model online &nbsp;·&nbsp; {str(device).upper()} &nbsp;·&nbsp; {MODEL_TAG}</span>
    <span style="color: var(--muted); font-size: 0.78rem; letter-spacing:0.08em;">ATTENTION-BASED CAPTIONING</span>
</div>
""", unsafe_allow_html=True)

with st.expander("Settings"):
    c1, c2 = st.columns(2)
    beam_size = c1.slider("Beam width (higher = better, slower)", 1, 5, 3)
    max_len = c2.slider("Maximum caption length", 8, 30, 20)
    lang_label = c1.selectbox("Narration language", list(LANGS))
    slow_speech = c2.checkbox("Slower speech", value=False)
    autoplay = c2.checkbox("Auto-play narration", value=True)

# ==========================================
# 7. TABS — INPUT
# ==========================================
tab1, tab2 = st.tabs([
    "  Live Camera  ",
    "  Upload Image  "
])

image_bytes = None

with tab1:
    st.markdown(
        f'<div class="section-title">{svg(ICON_CAMERA, size=16, color="#6ea8ff")} Capture from device</div>',
        unsafe_allow_html=True
    )
    camera_file = st.camera_input("Take a photo", label_visibility="collapsed")
    if camera_file:
        image_bytes = camera_file.getvalue()

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
        image_bytes = uploaded_file.getvalue()

# ==========================================
# 8. OUTPUT — IMAGE, CAPTION, CONFIDENCE, AUDIO
# ==========================================
if image_bytes:
    image = load_image(image_bytes)

    st.markdown(
        f'<div class="section-title">{svg(ICON_IMAGE, size=16, color="#6ea8ff")} Visual Scene</div>',
        unsafe_allow_html=True
    )
    st.image(image, **STRETCH)

    with st.spinner("Analyzing scene..."):
        hyps = run_captioning(image_bytes, beam_size, max_len)

    best = hyps[0]
    caption = pretty(best["text"]) or "No description could be generated for this image."
    level, pill_cls = confidence_level(best["confidence"]) if best["words"] else ("low", "pill-bad")

    st.markdown(f"""
    <div class="caption-card">
        <div class="caption-label">
            {svg(ICON_SPARK, size=14, color="#6ea8ff")}
            <span>Generated Caption</span>
        </div>
        <p class="caption-text">{html.escape(caption)}</p>
    </div>
    <span class="status-pill {pill_cls}">Confidence: {level} &nbsp;({best["confidence"] * 100:.0f}%)</span>
    """, unsafe_allow_html=True)

    if level == "low":
        st.warning("The model is not confident about this scene. Try better lighting, move closer to the "
                   "main object, or keep the camera steady. Don't rely on this description for anything safety-critical.")

    if len(hyps) > 1:
        with st.expander("Other possible descriptions"):
            for h in hyps[1:]:
                st.write(f"{pretty(h['text'])}  ·  {h['confidence'] * 100:.0f}%")

    if best["attention"] is not None and best["words"]:
        with st.expander("Where the model looked (per word)"):
            n = min(len(best["words"]), 12)
            st.image([attention_overlay(image, best["attention"][i]) for i in range(n)],
                     caption=best["words"][:n], width=120)

    # ---- narration --------------------------------------------------------
    st.markdown(
        f'<div class="section-title">{svg(ICON_VOLUME, size=16, color="#8affc1")} Voice Narration</div>',
        unsafe_allow_html=True
    )
    to_say = caption if level != "low" else "I am not very sure, but it looks like: " + caption
    audio, spoken, note = narrate(to_say, LANGS[lang_label], slow_speech)
    if note:
        st.info(note)
    if audio:
        if LANGS[lang_label] != "en" and not note:
            st.caption(spoken)
        replay = st.button("Replay Speech Description", **STRETCH)
        st.audio(audio, format="audio/mp3", autoplay=bool(autoplay or replay))

    # ---- session history ---------------------------------------------------
    key = hashlib.md5(image_bytes).hexdigest()
    hist = st.session_state.setdefault("history", [])
    if not hist or hist[0]["key"] != key:
        hist.insert(0, dict(key=key, text=caption, conf=best["confidence"]))
        del hist[8:]
    if len(hist) > 1:
        with st.expander("Recent descriptions"):
            for h in hist:
                st.write(f"{h['text']}  ·  {h['conf'] * 100:.0f}%")

st.markdown(
    '<div class="fine-print">Descriptions are generated by a small AI model and can be wrong. '
    'Confidence is the model\'s own estimate, not a guarantee.</div>',
    unsafe_allow_html=True
)
