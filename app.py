import streamlit as st
from PIL import Image

from matcher import HybridImageMatcher


st.set_page_config(page_title="Reference Plate Matcher", layout="wide")
st.title("Reference Plate Matcher")
st.write("Upload a reference plate and a property image to check whether the plate is present.")


@st.cache_resource
def load_matcher():
    return HybridImageMatcher()


reference_file = st.file_uploader("Reference plate image", type=["jpg", "jpeg", "png", "webp"])
property_file = st.file_uploader("Property image", type=["jpg", "jpeg", "png", "webp"])
prompt = st.text_input("Optional region hint", "plate . sign . house number .")

if reference_file and property_file:
    ref_img = Image.open(reference_file).convert("RGB")
    prop_img = Image.open(property_file).convert("RGB")
    left, right = st.columns(2)
    left.image(ref_img, caption="Reference plate", width="stretch")
    right.image(prop_img, caption="Property image", width="stretch")

    if st.button("Check plate", type="primary"):
        with st.spinner("Checking whether the reference plate is present…"):
            try:
                result = load_matcher().verify_presence(ref_img, prop_img, target_prompt=prompt)
            except Exception as exc:
                st.error(f"Could not analyze these images: {exc}")
                st.stop()

        if result.get("contains_reference"):
            st.success("ACCEPT — Reference plate is present in the property image.")
        else:
            st.error("REJECT — Reference plate is not present in the property image.")
