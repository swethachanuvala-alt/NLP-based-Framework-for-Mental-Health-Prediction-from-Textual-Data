"""
app.py
------
Streamlit app for the GloVe + GRU mental-health sentiment/intensity
classifier. Supports:
  - Single sentence prediction
  - Batch prediction from an uploaded CSV/XLSX file

Theme: animated night-sky / galaxy background (twinkling stars, a
softly glowing moon, drifting ringed planets). Native Streamlit
menu/header are left visible.

Run with:
    streamlit run app.py
"""

import pickle
import random
from string import Template

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing import preprocess_batch

MODEL_PATH = "saved_model/glove_gru.keras"
TOKENIZER_PATH = "saved_model/tokenizer.pkl"
LABEL_ENCODER_PATH = "saved_model/label_encoder.pkl"
MAX_LEN = 100  # must match the value used in train_glove_gru.py

# ---------------------------------------------------------------------
# Sentiment styling — one place to control color/emoji/label per class
# ---------------------------------------------------------------------
SENTIMENT_STYLE = {
    -2: {"label": "Very Negative", "emoji": "☄️", "color": "#FF6B6B", "glow": "rgba(255,107,107,0.35)"},
    -1: {"label": "Negative",      "emoji": "🌑", "color": "#F5A524", "glow": "rgba(245,165,36,0.35)"},
     0: {"label": "Neutral",       "emoji": "🌗", "color": "#9CA3D4", "glow": "rgba(156,163,212,0.35)"},
     1: {"label": "Positive",      "emoji": "🌟", "color": "#4ADE80", "glow": "rgba(74,222,128,0.35)"},
}
DEFAULT_STYLE = {"label": "Unknown", "emoji": "❔", "color": "#9CA3D4", "glow": "rgba(156,163,212,0.35)"}


def style_for(intensity: int) -> dict:
    return SENTIMENT_STYLE.get(int(intensity), DEFAULT_STYLE)


# ---------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder


def predict_texts(texts, model, tokenizer, label_encoder):
    """Run the full pipeline on a list of raw text strings.
    Returns (predicted_intensity, predicted_label, confidence, class_probabilities)
    """
    token_lists = preprocess_batch(texts)
    sequences = tokenizer.texts_to_sequences(token_lists)
    padded = pad_sequences(sequences, maxlen=MAX_LEN)

    probs = model.predict(padded, verbose=0)
    pred_idx = np.argmax(probs, axis=1)
    pred_intensity = label_encoder.inverse_transform(pred_idx)
    confidence = probs[np.arange(len(probs)), pred_idx]
    pred_label = [style_for(v)["label"] for v in pred_intensity]

    return pred_intensity, pred_label, confidence, probs


# ---------------------------------------------------------------------
# Starfield generator (classic pure-CSS multi-layer box-shadow trick)
# ---------------------------------------------------------------------
def _star_shadows(n, spread=2000, seed=None):
    rnd = random.Random(seed)
    return ", ".join(f"{rnd.randint(0, spread)}px {rnd.randint(0, spread)}px #FFF" for _ in range(n))


def inject_galaxy_background():
    small_stars = _star_shadows(500, 2000, seed=1)
    medium_stars = _star_shadows(150, 2000, seed=2)
    large_stars = _star_shadows(60, 2000, seed=3)

    css = Template(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

        .stApp {
            background: radial-gradient(ellipse at bottom, #0a0e2e 0%, #05061a 60%, #010108 100%);
            color: #EDEDF7;
            overflow-x: hidden;
        }

        /* keep the main content above the animated sky */
        .main .block-container { position: relative; z-index: 5; padding-top: 2rem; }

        /* ================= GALAXY BACKGROUND ================= */
        #galaxy-sky {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            overflow: hidden;
            z-index: 0;
            pointer-events: none;
        }

        #stars-small, #stars-medium, #stars-large {
            position: absolute;
            top: 0; left: 0;
            width: 1px; height: 1px;
            background: transparent;
        }
        #stars-small { box-shadow: $small_stars; animation: driftDown 140s linear infinite; }
        #stars-small::after {
            content: " "; position: absolute; top: 2000px; left: 0;
            width: 1px; height: 1px; box-shadow: $small_stars;
        }
        #stars-medium {
            box-shadow: $medium_stars; width: 2px; height: 2px;
            animation: driftDown 90s linear infinite, twinkle 4s ease-in-out infinite;
        }
        #stars-medium::after {
            content: " "; position: absolute; top: 2000px; left: 0;
            width: 2px; height: 2px; box-shadow: $medium_stars;
        }
        #stars-large {
            box-shadow: $large_stars; width: 3px; height: 3px; border-radius: 50%;
            animation: driftDown 60s linear infinite, twinkle 3s ease-in-out infinite;
        }
        #stars-large::after {
            content: " "; position: absolute; top: 2000px; left: 0;
            width: 3px; height: 3px; border-radius: 50%; box-shadow: $large_stars;
        }

        @keyframes driftDown {
            from { transform: translateY(0px); }
            to   { transform: translateY(2000px); }
        }
        @keyframes twinkle {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.35; }
        }

        /* ---- Moon ---- */
        #moon {
            position: absolute;
            top: 6%; right: 10%;
            width: 110px; height: 110px;
            border-radius: 50%;
            background: radial-gradient(circle at 35% 30%, #fdfcf7 0%, #e7e3d6 45%, #b9b6a8 100%);
            box-shadow:
                0 0 60px 12px rgba(253, 252, 247, 0.35),
                inset -14px -10px 0 0 rgba(0,0,0,0.10),
                inset 18px 22px 0 -6px rgba(160,158,148,0.5),
                inset -20px 14px 0 -10px rgba(160,158,148,0.4);
            animation: floatY 12s ease-in-out infinite;
        }
        @keyframes floatY {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(22px); }
        }

        /* ---- Planets ---- */
        .planet {
            position: absolute;
            border-radius: 50%;
            animation: orbitFloat 30s ease-in-out infinite;
        }
        .planet::after {
            content: "";
            position: absolute;
            top: 50%; left: 50%;
            width: 170%; height: 34%;
            border: 2px solid rgba(255,255,255,0.35);
            border-radius: 50%;
            transform: translate(-50%, -50%) rotate(-18deg);
        }
        #planet-1 {
            top: 62%; left: 8%;
            width: 70px; height: 70px;
            background: radial-gradient(circle at 35% 30%, #c084fc, #6d28d9 70%);
            box-shadow: 0 0 40px rgba(192,132,252,0.45);
            animation-duration: 26s;
        }
        #planet-2 {
            top: 18%; left: 6%;
            width: 34px; height: 34px;
            background: radial-gradient(circle at 35% 30%, #67e8f9, #0e7490 70%);
            box-shadow: 0 0 26px rgba(103,232,249,0.5);
            animation-duration: 20s;
            animation-delay: 2s;
        }
        #planet-3 {
            top: 74%; right: 12%;
            width: 46px; height: 46px;
            background: radial-gradient(circle at 35% 30%, #fca5a5, #b45309 70%);
            box-shadow: 0 0 30px rgba(252,165,165,0.4);
            animation-duration: 34s;
            animation-delay: 4s;
        }
        @keyframes orbitFloat {
            0%   { transform: translate(0px, 0px) rotate(0deg); }
            25%  { transform: translate(18px, -14px) rotate(4deg); }
            50%  { transform: translate(0px, -26px) rotate(0deg); }
            75%  { transform: translate(-18px, -14px) rotate(-4deg); }
            100% { transform: translate(0px, 0px) rotate(0deg); }
        }

        /* ---- Shooting stars ---- */
        .shooting-star {
            position: absolute;
            width: 2px; height: 2px;
            background: linear-gradient(-45deg, #fff, rgba(255,255,255,0));
            border-radius: 50%;
            filter: drop-shadow(0 0 6px #fff);
            animation: shoot 7s linear infinite;
            opacity: 0;
        }
        .shooting-star::before {
            content: "";
            position: absolute;
            top: 0; left: 0;
            width: 90px; height: 1px;
            background: linear-gradient(90deg, #fff, transparent);
            transform: translateX(-90px);
        }
        #shoot-1 { top: 12%; left: 70%; animation-delay: 0s; }
        #shoot-2 { top: 30%; left: 20%; animation-delay: 3.5s; }
        #shoot-3 { top: 55%; left: 85%; animation-delay: 6s; }
        @keyframes shoot {
            0% { opacity: 0; transform: translate(0, 0); }
            2% { opacity: 1; }
            15% { opacity: 0; transform: translate(-260px, 260px); }
            100% { opacity: 0; }
        }

        /* ================= UI CHROME ================= */
        .hero-wrap {
            padding: 34px 38px;
            border-radius: 22px;
            margin-bottom: 26px;
            background: linear-gradient(120deg, rgba(109,40,217,0.30), rgba(14,116,144,0.20));
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 8px 32px rgba(0,0,0,0.45);
            backdrop-filter: blur(6px);
        }
        .hero-title {
            font-size: 2.15rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(90deg, #C4B5FD, #67E8F9 60%, #FDE68A);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }
        .hero-sub { margin-top: 6px; color: #C9C9E3; font-size: 0.98rem; }
        .hero-badges { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }
        .badge {
            font-size: 0.75rem; font-weight: 600; padding: 5px 12px;
            border-radius: 999px; background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.16); color: #DCDCF5;
        }

        .glass-card {
            background: rgba(15,12,40,0.55);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 18px; padding: 22px 24px;
            backdrop-filter: blur(14px);
            box-shadow: 0 4px 24px rgba(0,0,0,0.35);
            margin-bottom: 18px;
        }

        .disclaimer {
            border-radius: 14px; padding: 12px 16px;
            background: rgba(103,232,249,0.08);
            border: 1px solid rgba(103,232,249,0.25);
            color: #C4F5EA; font-size: 0.86rem; margin-bottom: 22px;
        }

        .result-card {
            border-radius: 20px; padding: 26px;
            border: 1px solid var(--edge);
            background: linear-gradient(135deg, var(--glow), rgba(15,12,40,0.5));
            box-shadow: 0 0 40px var(--glow);
            display: flex; align-items: center; gap: 22px;
            backdrop-filter: blur(10px);
        }
        .result-emoji { font-size: 3rem; line-height: 1; }
        .result-label { font-size: 1.5rem; font-weight: 700; color: var(--edge); margin: 0; }
        .result-meta { color: #C9C9E3; font-size: 0.88rem; margin-top: 4px; }

        .gauge-wrap { display: flex; align-items: center; justify-content: center; }
        .gauge {
            width: 108px; height: 108px; border-radius: 50%;
            background: conic-gradient(var(--edge) calc(var(--pct) * 1%), rgba(255,255,255,0.10) 0);
            display: flex; align-items: center; justify-content: center;
        }
        .gauge-inner {
            width: 84px; height: 84px; border-radius: 50%;
            background: #0a0820;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
        }
        .gauge-pct { font-weight: 700; font-size: 1.05rem; color: #F2F2FA; }
        .gauge-caption { font-size: 0.65rem; color: #9C9CC0; margin-top: -2px; }

        .prob-row { display: flex; align-items: center; gap: 10px; margin: 9px 0; }
        .prob-label { width: 130px; font-size: 0.85rem; color: #DCDCF5; display: flex; align-items: center; gap: 6px; }
        .prob-track { flex: 1; height: 10px; border-radius: 999px; background: rgba(255,255,255,0.08); overflow: hidden; }
        .prob-fill { height: 100%; border-radius: 999px; }
        .prob-val { width: 48px; text-align: right; font-size: 0.8rem; color: #9C9CC0; font-family: 'JetBrains Mono', monospace; }

        .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 14px 0 4px; }
        .stat-chip {
            border-radius: 16px; padding: 14px 16px;
            border: 1px solid var(--edge2);
            background: linear-gradient(135deg, var(--glow2), rgba(15,12,40,0.5));
            backdrop-filter: blur(8px);
        }
        .stat-chip .n { font-size: 1.6rem; font-weight: 800; color: var(--edge2); }
        .stat-chip .l { font-size: 0.78rem; color: #C9C9E3; margin-top: 2px; }

        .section-title {
            font-size: 1.05rem; font-weight: 700; color: #F2F2FA;
            display: flex; align-items: center; gap: 8px; margin-bottom: 10px;
        }

        .legend-row { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 4px; }
        .legend-chip { display: flex; align-items: center; gap: 6px; font-size: 0.82rem; color: #DCDCF5; }
        .legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

        /* Streamlit widgets */
        .stTextArea textarea {
            background: rgba(255,255,255,0.06) !important;
            color: #F2F2FA !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
        }
        .stTextArea textarea::placeholder { color: #8888aa !important; }

        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(255,255,255,0.05) !important;
            border: 1.5px dashed rgba(196,181,253,0.5) !important;
            border-radius: 16px !important;
        }

        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(90deg, #6D28D9, #0E7490) !important;
            color: white !important;
            border: none !important;
            border-radius: 999px !important;
            padding: 0.55rem 1.6rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 18px rgba(109,40,217,0.4) !important;
            transition: transform 0.15s ease !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px) scale(1.015);
        }

        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.05);
            border-radius: 12px 12px 0 0;
            color: #C9C9E3;
            padding: 8px 18px;
            border: 1px solid rgba(255,255,255,0.08);
            border-bottom: none;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, rgba(109,40,217,0.35), rgba(14,116,144,0.30)) !important;
            color: #FFFFFF !important;
        }

        div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
        div[data-testid="stExpander"] {
            background: rgba(15,12,40,0.45);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 14px;
        }

        code { color: #67E8F9; }
        </style>

        <div id="galaxy-sky">
            <div id="stars-small"></div>
            <div id="stars-medium"></div>
            <div id="stars-large"></div>
            <div id="moon"></div>
            <div class="planet" id="planet-1"></div>
            <div class="planet" id="planet-2"></div>
            <div class="planet" id="planet-3"></div>
            <div class="shooting-star" id="shoot-1"></div>
            <div class="shooting-star" id="shoot-2"></div>
            <div class="shooting-star" id="shoot-3"></div>
        </div>
        """
    ).substitute(small_stars=small_stars, medium_stars=medium_stars, large_stars=large_stars)

    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# UI helper renderers
# ---------------------------------------------------------------------
def render_hero():
    legend_items = "".join(
        f"<span class='legend-chip'><span class='legend-dot' "
        f"style='background:{style_for(v)['color']}'></span>{style_for(v)['emoji']} {style_for(v)['label']}</span>"
        for v in [1, 0, -1, -2]
    )
    st.markdown(
        f"""
        <div class="hero-wrap">
            <p class="hero-title" style="font-size:1.6rem; font-weight:500; -webkit-text-fill-color:#FFFFFF; background:none;">
            <b>MindPlus-NLP based Framework for Mental Health Prediction from Textual Data</b>
            </p>
            <p class="hero-sub">Understand the emotional intensity behind mental-health posts, powered by GloVe embeddings + a GRU network.</p>
            <div class="hero-badges">
                <span class="badge">🔤 GloVe 100d</span>
                <span class="badge">🔁 GRU · 128 units</span>
                <span class="badge">⚡ Real-time inference</span>
            </div>
            <div class="section-title" style="margin-top:18px;">🎨 Sentiment legend</div>
            <div class="legend-row">
                {legend_items}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(intensity, label, confidence):
    s = style_for(intensity)
    pct = round(float(confidence) * 100, 1)
    st.markdown(
        f"""
        <div class="result-card" style="--edge:{s['color']}; --glow:{s['glow']};">
            <div class="result-emoji">{s['emoji']}</div>
            <div style="flex:1;">
                <p class="result-label">{s['label']}</p>
                <p class="result-meta">Predicted intensity score: <code>{int(intensity)}</code></p>
            </div>
            <div class="gauge-wrap">
                <div class="gauge" style="--edge:{s['color']}; --pct:{pct};">
                    <div class="gauge-inner">
                        <div class="gauge-pct">{pct}%</div>
                        <div class="gauge-caption">confidence</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer():
    st.markdown(
        """
        <div class="disclaimer">
        ℹ️ This is a coursework / demo NLP model — <b>not</b> a diagnostic or clinical tool.
        If you or someone you know is struggling, please reach out to a mental health
        professional or a local helpline.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="MindPlus — Sentiment Intelligence", page_icon="🌌", layout="centered")
    inject_galaxy_background()
    render_hero()

    try:
        model, tokenizer, label_encoder = load_artifacts()
    except (OSError, IOError, FileNotFoundError) as e:
        st.error(
            "Couldn't find the trained model artifacts. Run `python train_glove_gru.py` "
            "first to generate saved_model/glove_gru.keras, tokenizer.pkl and "
            "label_encoder.pkl.\n\n"
            f"Details: {e}"
        )
        st.stop()

    with st.expander("🧭 About this model", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                "- **Embeddings:** GloVe (100d, frozen)\n"
                "- **Sequence model:** GRU · 128 units\n"
                "- **Head:** Dense(64, relu) → Dropout(0.5) → Softmax"
            )
        with col_b:
            st.markdown(
                "- **Target:** post `intensity` (-2 → 1)\n"
                "- **Max sequence length:** 100 tokens\n"
                "- **Class balancing:** RandomOverSampler on training set"
            )

    tab_single, tab_batch = st.tabs(["✍️  Single Sentence", "📄  Batch (CSV / Excel)"])

    # ---------------- Single sentence ----------------
    with tab_single:
        st.markdown('<div class="section-title">✍️ Analyze a single post</div>', unsafe_allow_html=True)
        text_input = st.text_area(
            "Enter a sentence or post",
            height=150,
            placeholder="e.g. I've been feeling really overwhelmed and hopeless lately...",
            label_visibility="collapsed",
        )

        if st.button("✨ Predict sentiment", type="primary", key="predict_single"):
            if not text_input.strip():
                st.warning("Please enter some text first.")
            else:
                with st.spinner("Reading between the lines..."):
                    intensity, label, confidence, probs = predict_texts(
                        [text_input], model, tokenizer, label_encoder
                    )
                render_result_card(intensity[0], label[0], confidence[0])
                render_disclaimer()

    # ---------------- Batch prediction ----------------
    with tab_batch:
        st.markdown('<div class="section-title">📄 Analyze a batch of posts</div>', unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#C9C9E3;font-size:0.9rem;'>Upload a CSV or Excel file with a text column "
            "(named <code>posts</code> or <code>text</code>). Every row will be classified.</p>",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    batch_df = pd.read_csv(uploaded_file)
                else:
                    batch_df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Couldn't read that file: {e}")
                st.stop()

            text_col = None
            for candidate in ["posts", "text", "post", "sentence", "review"]:
                if candidate in batch_df.columns:
                    text_col = candidate
                    break

            if text_col is None:
                st.warning("No obvious text column found. Pick one below:")
                text_col = st.selectbox("Text column", batch_df.columns.tolist())

            st.markdown(f"<p style='color:#9C9CC0;font-size:0.85rem;'>Using column <code>{text_col}</code> · {len(batch_df)} rows</p>", unsafe_allow_html=True)
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("🚀 Run batch prediction", type="primary", key="predict_batch"):
                clean_df = batch_df.dropna(subset=[text_col]).copy()
                texts = clean_df[text_col].astype(str).tolist()

                with st.spinner(f"Predicting {len(texts)} rows..."):
                    intensity, label, confidence, _ = predict_texts(
                        texts, model, tokenizer, label_encoder
                    )

                clean_df["predicted_intensity"] = intensity
                clean_df["predicted_sentiment"] = label
                clean_df["confidence"] = np.round(confidence, 4)

                render_disclaimer()

                st.markdown('<div class="section-title" style="margin-top:18px;">🗂️ Results</div>', unsafe_allow_html=True)
                st.dataframe(clean_df, use_container_width=True)

                csv_bytes = clean_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️  Download predictions as CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()