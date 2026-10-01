import re
import string
import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st


# ---------------------------------------------------------------------------
# The pickled TfidfVectorizer references `wordopt` as its preprocessor, so this
# function MUST exist under this exact name and be visible in __main__ before
# the model is loaded. If your training notebook used a different cleaning
# routine, replace the body below with the original one.
# ---------------------------------------------------------------------------
def wordopt(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


# Make the function discoverable to pickle no matter how Streamlit runs the script
sys.modules["__main__"].wordopt = wordopt

MODEL_PATH = Path(__file__).parent / "news.pkl"
LABELS = {0: "Fake News", 1: "Real News"}
MIN_WORDS = 20

SAMPLES = {
    "Sample: policy report": (
        "The finance ministry said on Tuesday that quarterly tax collections rose "
        "by four percent compared with the same period last year, according to a "
        "statement released after a meeting with state officials. Analysts noted "
        "that the increase was driven mainly by higher corporate payments, while "
        "the government said it would publish a full breakdown next month."
    ),
    "Sample: sensational claim": (
        "SHOCKING!!! You won't believe what they are hiding from you! Insiders "
        "reveal that a secret group controls everything and the mainstream media "
        "refuses to tell the truth. Share this before it gets deleted, because "
        "they do not want you to know what is really going on behind the scenes."
    ),
}


# ------------------------------- Model -------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model = joblib.load(MODEL_PATH)
    # Compatibility: the model was saved with a newer scikit-learn that dropped
    # `multi_class`. Older versions still read it, so restore it if missing.
    # (For a 2-class model, "auto" behaves the same as the newer code.)
    clf = model.named_steps["classifier"]
    if not hasattr(clf, "multi_class"):
        clf.multi_class = "auto"
    return model


def top_words(model, text, n=8):
    """Words in the article that pushed the prediction most in each direction."""
    tfidf = model.named_steps["Tfidf"]
    clf = model.named_steps["classifier"]
    vec = tfidf.transform([text])
    names = tfidf.get_feature_names_out()
    idx = vec.nonzero()[1]
    if len(idx) == 0:
        return [], []
    contrib = vec[0, idx].toarray().ravel() * clf.coef_[0][idx]
    order = np.argsort(contrib)
    fake = [(names[idx[i]], contrib[i]) for i in order[:n] if contrib[i] < 0]
    real = [(names[idx[i]], contrib[i]) for i in order[::-1][:n] if contrib[i] > 0]
    return fake, real


# ------------------------------- Styling -----------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 780px; padding-top: 2.2rem; padding-bottom: 3rem; }

.hero h1 {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 2.6rem; line-height: 1.1; margin: 0 0 .4rem 0; font-weight: 700;
    letter-spacing: -0.01em;
}
.hero p { font-size: 1.05rem; opacity: .75; margin: 0 0 1.4rem 0; max-width: 560px; }

.verdict {
    border-radius: 16px; padding: 1.4rem 1.6rem; margin: 1.2rem 0 1rem 0;
    border: 1px solid; display: flex; align-items: center; gap: 1.2rem;
}
.verdict.real { background: rgba(22,163,74,.10); border-color: rgba(22,163,74,.45); }
.verdict.fake { background: rgba(220,38,38,.10); border-color: rgba(220,38,38,.45); }
.verdict .icon { font-size: 2.4rem; line-height: 1; }
.verdict .title { font-family: 'Fraunces', Georgia, serif; font-size: 1.7rem; font-weight: 700; margin: 0; }
.verdict .sub { margin: .15rem 0 0 0; opacity: .8; font-size: .95rem; }

.meter { margin: .6rem 0 1.2rem 0; }
.meter-bar {
    height: 14px; border-radius: 999px; overflow: hidden; display: flex;
    background: rgba(128,128,128,.2);
}
.meter-bar .f { background: #dc2626; }
.meter-bar .r { background: #16a34a; }
.meter-legend { display: flex; justify-content: space-between; font-size: .85rem; margin-top: .4rem; opacity: .85; }

.chip {
    display: inline-block; padding: .22rem .7rem; margin: 0 .35rem .4rem 0;
    border-radius: 999px; font-size: .88rem; font-weight: 500; border: 1px solid;
}
.chip.fake { background: rgba(220,38,38,.10); border-color: rgba(220,38,38,.4); }
.chip.real { background: rgba(22,163,74,.10); border-color: rgba(22,163,74,.4); }
.hint { opacity: .65; font-size: .85rem; margin-top: .2rem; }

div.stButton > button[kind="primary"] { font-weight: 600; border-radius: 10px; padding: .55rem 1.2rem; }
div.stButton > button { border-radius: 10px; }
textarea { font-size: .98rem !important; line-height: 1.55 !important; }
</style>
"""


def chips(words, kind):
    if not words:
        return "<span class='hint'>No strong signals found.</span>"
    return "".join(f"<span class='chip {kind}'>{w}</span>" for w, _ in words)


def set_sample(key):
    st.session_state["article"] = SAMPLES[key]


def clear_text():
    st.session_state["article"] = ""


# ------------------------------- UI ----------------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

model = load_model()
classes = list(model.classes_)

with st.sidebar:
    st.header("About")
    st.write(
        "This tool uses TF-IDF features and a Logistic Regression classifier "
        "to judge an article by its writing patterns."
    )
    st.info("It does not check facts. Always confirm important claims with a trusted source.")
    show_words = st.toggle("Show influential words", value=True)
    st.divider()
    st.subheader("Tips")
    st.markdown(
        "- Paste the headline **and** the body.\n"
        "- Longer articles give steadier results.\n"
        "- Remove ads or menu text if you copied a full web page."
    )

st.markdown(
    """
    <div class="hero">
        <h1>📰 Fake News Detector</h1>
        <p>Paste a news article and see whether its writing style looks more like
        real reporting or fake news.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "article" not in st.session_state:
    st.session_state["article"] = ""

# Quick-start samples
st.caption("No article handy? Try an example:")
s1, s2, _ = st.columns([1.2, 1.4, 1.4])
s1.button("Sample: policy report", on_click=set_sample, args=("Sample: policy report",), use_container_width=True)
s2.button("Sample: sensational claim", on_click=set_sample, args=("Sample: sensational claim",), use_container_width=True)

text = st.text_area(
    "Article text",
    key="article",
    height=260,
    placeholder="Paste the headline and body of the article here...",
    label_visibility="collapsed",
)

word_count = len(text.split())
info_col, clear_col, go_col = st.columns([2.2, 1, 1.4])
with info_col:
    if word_count == 0:
        st.markdown("<div class='hint'>Waiting for text</div>", unsafe_allow_html=True)
    elif word_count < MIN_WORDS:
        st.markdown(
            f"<div class='hint'>{word_count} words. Add {MIN_WORDS - word_count} more for a reliable result.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"<div class='hint'>{word_count} words. Ready to analyze.</div>", unsafe_allow_html=True)
clear_col.button("Clear", on_click=clear_text, use_container_width=True)
analyze = go_col.button("Analyze article", type="primary", use_container_width=True)

if analyze:
    if word_count < MIN_WORDS:
        st.warning(f"Please enter at least {MIN_WORDS} words so the model has enough to work with.")
    else:
        with st.spinner("Reading the article..."):
            proba = model.predict_proba([text])[0]
            pred = int(classes[int(np.argmax(proba))])
            confidence = float(proba.max())
            p_fake = float(proba[classes.index(0)])
            p_real = float(proba[classes.index(1)])

        is_real = pred == 1
        if confidence >= 0.85:
            strength = "Strong signal"
        elif confidence >= 0.65:
            strength = "Moderate signal"
        else:
            strength = "Weak signal, treat with caution"

        st.markdown(
            f"""
            <div class="verdict {'real' if is_real else 'fake'}">
                <div class="icon">{'✅' if is_real else '🚨'}</div>
                <div>
                    <p class="title">{LABELS[pred]}</p>
                    <p class="sub">{confidence:.1%} confidence · {strength}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="meter">
                <div class="meter-bar">
                    <div class="f" style="width:{p_fake*100:.1f}%"></div>
                    <div class="r" style="width:{p_real*100:.1f}%"></div>
                </div>
                <div class="meter-legend">
                    <span>Fake {p_fake:.1%}</span>
                    <span>Real {p_real:.1%}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if show_words:
            fake_w, real_w = top_words(model, text)
            with st.expander("Which words influenced this result?", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Pushing toward fake**")
                    st.markdown(chips(fake_w, "fake"), unsafe_allow_html=True)
                with c2:
                    st.markdown("**Pushing toward real**")
                    st.markdown(chips(real_w, "real"), unsafe_allow_html=True)
                st.markdown(
                    "<div class='hint'>Words are ranked by how strongly they affected the model's decision.</div>",
                    unsafe_allow_html=True,
                )

        st.caption("This is a statistical estimate based on writing style, not a fact-check.")