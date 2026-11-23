import streamlit as st
import sys
import tempfile
from pathlib import Path
from PIL import Image, ImageOps
from huggingface_hub import snapshot_download

# ==========================================
# DEEPGUARD - SMAUG DETECTOR
# ==========================================

st.set_page_config(
    page_title="DeepGuard AI",
    page_icon="🛡️",
    layout="wide"
)

MODEL_ID = "abacusai/Smaug-AI-Image-Detector"

@st.cache_resource
def load_detector():

    model_path = snapshot_download(MODEL_ID)

    sys.path.insert(0, model_path)

    from smaug_image_detect import SmaugImageDetect

    detector = SmaugImageDetect.from_pretrained(model_path)

    return detector


st.title("🛡️ DeepGuard AI")
st.subheader("AI-Based Synthetic Image Detection")

st.write(
    "Upload an image to analyze whether it is likely to be "
    "AI-generated or a real photograph."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    try:

        with st.spinner("Analyzing image..."):

            detector = load_detector()

            # Save the uploaded image temporarily
            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False
            ) as temp_file:

                temp_path = temp_file.name

                image.save(temp_path, format="JPEG")

            # Run the tested detector
            score = detector(temp_path)

            # Remove temporary image
            Path(temp_path).unlink(missing_ok=True)

        ai_score = float(score)
        real_score = 1.0 - ai_score

        st.subheader("Detection Results")

        st.write(f"AI-generated score: {ai_score * 100:.2f}%")
        st.progress(max(0.0, min(1.0, ai_score)))

        st.write(f"Real-image score: {real_score * 100:.2f}%")
        st.progress(max(0.0, min(1.0, real_score)))

        if ai_score >= 0.80:

            st.error("🚨 LIKELY AI-GENERATED")

        elif ai_score <= 0.20:

            st.success("✅ LIKELY REAL")

        else:

            st.warning("⚠️ INCONCLUSIVE")

        st.caption(
            "Experimental AI image detector. Scores are model outputs, "
            "not guaranteed probabilities of authenticity."
        )

    except Exception as e:

        st.error(f"Analysis error: {e}")

st.divider()

st.caption(
    "DeepGuard | AI Image Detection Prototype"
)