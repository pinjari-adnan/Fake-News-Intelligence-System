import streamlit as st
import pickle
import re
import html

st.set_page_config(
    page_title="AI Fake News Intelligence System",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
/* Background */
.stApp{
    background: radial-gradient(circle at 15% 10%, rgba(124,58,237,0.18) 0%, rgba(0,0,0,0) 35%),
                radial-gradient(circle at 85% 30%, rgba(255,75,110,0.14) 0%, rgba(0,0,0,0) 40%),
                linear-gradient(135deg,#0b1220,#0f172a,#111827);
    color: #e5e7eb;
}
.block-container{ padding-top: 1.2rem; }

/* Remove toolbar/top decorations */
header, footer {visibility: hidden;}
div[data-testid="stDecoration"] {display:none;}
div[data-testid="stToolbar"] {display:none;}
div[data-testid="stStatusWidget"] {display:none;}

/* ✅ REMOVE that empty long bar under title */
div[data-testid="stHorizontalBlock"] > div:has(div[aria-label=""]) {display:none !important;}
/* Sometimes Streamlit renders empty blocks - hide very small empty blocks */
div[data-testid="stHorizontalBlock"] div:empty {display:none !important;}

/* Title */
.title{
    font-size: 40px;
    font-weight: 900;
    margin: 0 0 10px 0;
    background: linear-gradient(90deg,#ff4b6e,#ff8c42,#7c3aed);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

/* Cards */
.card{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 16px 16px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.30);
}
.section-title{
    font-size: 18px;
    font-weight: 900;
    margin: 0 0 10px 0;
    color: #ffffff;
}

/* Button */
.stButton>button{
    background: linear-gradient(90deg,#ff4b6e,#ff8c42) !important;
    color: white !important;
    border: 0 !important;
    border-radius: 12px !important;
    font-weight: 900 !important;
    padding: 10px 16px !important;
}

/* Metrics */
[data-testid="stMetricValue"]{
    color: #ffffff !important;
    font-size: 28px !important;
    font-weight: 900 !important;
}
[data-testid="stMetricLabel"]{
    color: #e5e7eb !important;
    font-weight: 800 !important;
}

/* Badge */
.badge{
    display:inline-flex;
    gap:8px;
    align-items:center;
    padding:10px 14px;
    border-radius:999px;
    font-weight:900;
    border:1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.06);
}
.badge-real{ border-color: rgba(34,197,94,0.35); }
.badge-fake{ border-color: rgba(239,68,68,0.35); }

/* Progress */
div[role="progressbar"]{
    height: 10px !important;
    border-radius: 999px !important;
}

/* Highlight box */
.highlight-box{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 14px;
    line-height: 1.75;
    font-size: 15px;
    color: #ffffff;
    white-space: pre-wrap;
}
.hl-real{
    background: rgba(34,197,94,0.28);
    border-bottom: 2px solid rgba(34,197,94,0.80);
    padding: 1px 4px;
    border-radius: 6px;
}
.hl-fake{
    background: rgba(239,68,68,0.28);
    border-bottom: 2px solid rgba(239,68,68,0.80);
    padding: 1px 4px;
    border-radius: 6px;
}

/* Explain chips */
.chips{ display:flex; flex-wrap:wrap; gap:8px; }
.chip{
    padding:8px 10px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 900;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.06);
    color:#fff;
}
.chip-real{ border-color: rgba(34,197,94,0.35); }
.chip-fake{ border-color: rgba(239,68,68,0.35); }

/* ✅ SIDEBAR FIX (dark bg + bold white text) */
section[data-testid="stSidebar"]{
    background: rgba(2,6,23,0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.10);
}
section[data-testid="stSidebar"] *{
    color: #ffffff !important;
    font-weight: 900 !important;
    font-size: 14px !important;
}
section[data-testid="stSidebar"] label{
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------
# HELPERS
# -----------------------------
def explain_prediction(text: str, top_k: int = 10):
    transformed = vectorizer.transform([text])
    if transformed.nnz == 0:
        return []
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    idxs = transformed.nonzero()[1]
    word_scores = [(feature_names[i], float(coefs[i])) for i in idxs]
    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return word_scores[:top_k]

def highlight_text(raw_text: str, top_words):
    safe = html.escape(raw_text)
    words_sorted = sorted({w for w, _ in top_words if len(w) >= 3}, key=len, reverse=True)
    sign = {w: ("fake" if s < 0 else "real") for w, s in top_words}
    for w in words_sorted:
        css = "hl-fake" if sign.get(w) == "fake" else "hl-real"
        pattern = re.compile(rf"(?i)\b{re.escape(w)}\b")
        safe = pattern.sub(lambda m: f"<span class='{css}'>{m.group(0)}</span>", safe)
    return safe

def risk_from_fake_prob(fake_prob_pct: float):
    if fake_prob_pct >= 80:
        return "HIGH"
    elif fake_prob_pct >= 50:
        return "MEDIUM"
    return "LOW"

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<div class='title'>🧠 AI Fake News Intelligence System</div>", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR CONTROLS (now bold + white)
# -----------------------------
st.sidebar.title("⚙️ Controls")
explain_mode = st.sidebar.checkbox("Explainable AI (Top words)", value=True)
highlight_mode = st.sidebar.checkbox("Highlight Mode (Mark words)", value=True)
show_probs = st.sidebar.checkbox("Show Probabilities", value=False)

# -----------------------------
# MAIN UI
# -----------------------------
left, right = st.columns([1.35, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📝 Paste News Article / Headline</div>", unsafe_allow_html=True)
    news_text = st.text_area("", height=170, placeholder="Paste full news here (2–3 lines better).")
    analyze = st.button("🚀 Analyze")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📌 Quick Stats</div>", unsafe_allow_html=True)
    wc = len(news_text.split()) if news_text else 0
    chars = len(news_text) if news_text else 0
    a, b = st.columns(2)
    with a:
        st.metric("Word Count", wc)
    with b:
        st.metric("Characters", chars)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# PREDICTION
# -----------------------------
if analyze:
    if not news_text or news_text.strip() == "":
        st.warning("⚠️ Please paste some news text first.")
    else:
        X = vectorizer.transform([news_text])
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]

        fake_prob = float(proba[0]) * 100
        real_prob = float(proba[1]) * 100
        confidence = max(fake_prob, real_prob)

        risk = risk_from_fake_prob(fake_prob)
        top_words = explain_prediction(news_text, top_k=10)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Analysis Result</div>", unsafe_allow_html=True)

        if pred == 0:
            st.markdown("<div class='badge badge-fake'>🚨 FAKE</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='badge badge-real'>✅ REAL</div>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Confidence", f"{confidence:.2f}%")
        with m2:
            st.metric("Fake Prob", f"{fake_prob:.2f}%")
        with m3:
            st.metric("Real Prob", f"{real_prob:.2f}%")
        with m4:
            st.metric("Fake Risk", risk)

        st.progress(int(min(max(confidence, 0), 100)))

        if show_probs:
            st.write({"fake_prob": fake_prob/100, "real_prob": real_prob/100})

        if highlight_mode:
            st.markdown("<div class='section-title'>🧾 Highlighted Text</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='highlight-box'>{highlight_text(news_text, top_words)}</div>", unsafe_allow_html=True)

        if explain_mode:
            st.markdown("<div class='section-title'>🔎 Explainable AI (Top words)</div>", unsafe_allow_html=True)
            if not top_words:
                st.info("Not enough known words found. Try a longer news text.")
            else:
                st.markdown("<div class='chips'>", unsafe_allow_html=True)
                for w, s in top_words:
                    if s < 0:
                        st.markdown(f"<div class='chip chip-fake'>🔴 {w} • FAKE</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='chip chip-real'>🟢 {w} • REAL</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
