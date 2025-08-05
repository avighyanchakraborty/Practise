# Reference only: Full app.py file is not shared publicly.
# View the demo or read-only content here:
# https://drive.google.com/file/d/1ZIXolxtXFkHbOR277xlHHdnbsLpuQKw0/view?usp=sharing

# app.py (DEMO VERSION)
# Full implementation is private

import streamlit as st

st.set_page_config(layout="wide")
st.title("🌿 Leaf Disease Detection and Severity Estimation")

st.markdown("""
This is a demo of a leaf disease detection app.  
The full version includes:
- Background removal using `rembg`
- Deep learning models for leaf type and disease detection (`.h5` files)
- Image analysis using OpenCV and GLCM (texture features)
- Visualization of disease masks and severity %  
""")

st.info("⚠️ This demo does not include model files or prediction logic.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.success("✅ Image uploaded. (Prediction logic not included in demo)")
    st.subheader("🔍 Prediction Results")
    st.write("**Leaf Type**: *DemoLeaf*")
    st.write("**Disease**: *DemoDisease*")
    
    st.subheader("📊 Disease Severity & Texture")
    st.write("**Diseased Area**: 15.5%")
    st.write("**Contrast**: 1.203")
    st.write("**Homogeneity**: 0.832")
    st.write("**Energy**: 0.653")
    st.write("**Correlation**: 0.912")

    st.subheader("🖼️ Visualization (Sample)")
    col1, col2, col3, col4 = st.columns(4)
    col1.image(uploaded_file, caption="Original", use_column_width=True)
    col2.image(uploaded_file, caption="No Background", use_column_width=True)
    col3.image(uploaded_file, caption="Green Mask", use_column_width=True)
    col4.image(uploaded_file, caption="Diseased Mask", use_column_width=True)

