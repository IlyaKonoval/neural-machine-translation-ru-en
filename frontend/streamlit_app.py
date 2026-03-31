import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RU → EN Translator", page_icon="🌐", layout="centered")

st.title("Transformer Translator")
st.markdown("**Russian → English** neural machine translation")

with st.sidebar:
    st.header("Settings")
    beam_size = st.slider("Beam size", min_value=1, max_value=10, value=5)

    st.divider()
    st.subheader("Model Info")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        if health["model_loaded"]:
            st.success(f"Model loaded on **{health['device']}**")
        else:
            st.error("Model not loaded")
    except requests.exceptions.ConnectionError:
        st.error("API unavailable. Start with:\n`uvicorn api.app:app`")

    st.divider()
    st.subheader("Architecture")
    st.markdown("""
    - Custom Transformer (from scratch)
    - Multi-Head Attention (8 heads)
    - 4 Encoder + 4 Decoder layers
    - BERT tokenizer
    - Beam Search decoding
    - Trained on 363K RU-EN pairs
    """)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Russian")
    source_text = st.text_area(
        "Enter text",
        height=150,
        placeholder="Привет, как дела?",
        label_visibility="collapsed",
    )

with col2:
    st.subheader("English")
    translation_placeholder = st.empty()

translate_btn = st.button("Translate", type="primary", use_container_width=True)

if translate_btn and source_text.strip():
    try:
        with st.spinner("Translating..."):
            resp = requests.post(
                f"{API_URL}/translate",
                json={"text": source_text, "beam_size": beam_size},
                timeout=30,
            )
        if resp.status_code == 200:
            data = resp.json()
            translation_placeholder.text_area(
                "Translation",
                value=data["translation"],
                height=150,
                label_visibility="collapsed",
                disabled=True,
            )
            st.caption(f"Beam size: {data['beam_size']} | Time: {data['elapsed_ms']:.0f} ms")
        else:
            st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API.")
elif translate_btn:
    st.warning("Please enter text to translate.")

st.divider()
st.subheader("Examples")
examples = [
    "Привет, как дела?",
    "Я люблю программирование.",
    "Сегодня хорошая погода.",
    "Как тебя зовут?",
    "Спасибо большое!",
]

cols = st.columns(len(examples))
for i, example in enumerate(examples):
    with cols[i]:
        if st.button(example, key=f"example_{i}", use_container_width=True):
            st.session_state["_rerun_text"] = example
            st.rerun()

if "_rerun_text" in st.session_state:
    source_text = st.session_state.pop("_rerun_text")
    try:
        resp = requests.post(
            f"{API_URL}/translate",
            json={"text": source_text, "beam_size": beam_size},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            st.info(f"**RU:** {source_text}")
            st.success(f"**EN:** {data['translation']}")
            st.caption(f"Time: {data['elapsed_ms']:.0f} ms")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API.")
