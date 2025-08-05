import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import rembg
from tensorflow.keras.models import load_model
from ultralyticsplus import YOLO
from skimage.feature import graycomatrix, graycoprops

# ---------------------- Helper Functions ----------------------

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image_class(model_path, image_path):
    model = load_model(model_path)
    processed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    return predicted_class

def remove_background(image_path):
    img = Image.open(image_path).convert('RGB')
    input_image = np.array(img)
    output_image = rembg.remove(img)
    output_image = Image.fromarray(np.array(output_image)).convert('RGB')
    return input_image, np.array(output_image)

def get_color_ranges(leaf_type):
    if leaf_type == "Apple":
        green_range = ((25, 40, 20), (100, 255, 100))
        brown_range = ((60, 40, 20), (180, 100, 80))
    elif leaf_type == "Mango":
        green_range = ((20, 60, 20), (90, 255, 90))
        brown_range = ((50, 30, 10), (160, 110, 70))
    elif leaf_type == "Cotton":
        green_range = ((20, 60, 20), (90, 255, 90))
        brown_range = ((50, 30, 10), (160, 110, 70))
    elif leaf_type == "Grape":
        green_range = ((15, 50, 20), (80, 255, 90))
        brown_range = ((40, 25, 10), (160, 100, 70))
    else:
        green_range = ((25, 40, 20), (100, 255, 100))
        brown_range = ((60, 40, 20), (180, 100, 80))
    return {'green': green_range, 'brown': brown_range}

def analyze_leaf(image, color_ranges):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    green_lower = np.array(color_ranges['green'][0], dtype=np.uint8)
    green_upper = np.array(color_ranges['green'][1], dtype=np.uint8)
    brown_lower = np.array(color_ranges['brown'][0], dtype=np.uint8)
    brown_upper = np.array(color_ranges['brown'][1], dtype=np.uint8)

    green_mask = cv2.inRange(image, green_lower, green_upper)
    disease_mask = cv2.inRange(image, brown_lower, brown_upper)

    green_area = cv2.countNonZero(green_mask)
    disease_area = cv2.countNonZero(disease_mask)
    total_area = green_area + disease_area
    disease_percentage = (disease_area / total_area) * 100 if total_area else 0

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    return green_mask, disease_mask, disease_percentage, {
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation
    }

def visualize_results(input_image, output_image, green_mask, disease_mask):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(input_image)
    axs[0].set_title("Original Image")
    axs[1].imshow(output_image)
    axs[1].set_title("Background Removed")
    axs[2].imshow(green_mask, cmap='gray')
    axs[2].set_title("Green Mask")
    axs[3].imshow(disease_mask, cmap='gray')
    axs[3].set_title("Diseased Mask")
    for ax in axs:
        ax.axis('off')
    st.pyplot(fig)

def predict_leaf_and_disease(image_path):
    leaf_type_index = predict_image_class('leaf_type.h5', image_path)
    leaf_types = ['Apple', 'Cotton', 'Grape', 'Mango']
    leaf_type = leaf_types[leaf_type_index]

    disease_model_paths = {
        'Apple': 'model_leaf_apple.h5',
        'Cotton': 'model_leaf_cotton.h5',
        'Grape': 'model_leaf_grape.h5',
        'Mango': 'model_leaf_mango.h5'
    }

    model_path = disease_model_paths.get(leaf_type)
    disease_index = predict_image_class(model_path, image_path)

    disease_classes = {
        'Apple': ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy'],
        'Cotton': ['diseased cotton leaf', 'fresh cotton leaf'],
        'Grape': ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy'],
        'Mango': ['Anthracnose', 'Bacterial_Canker', 'Cutting_Weevil', 'Gall_Midge', 'Powdery_Mildew', 'Sooty_Mould']
    }

    disease_labels = disease_classes.get(leaf_type, [])
    disease = disease_labels[disease_index] if disease_index < len(disease_labels) else "Unknown"

    return leaf_type, disease

# ---------------------- Streamlit UI ----------------------

st.set_page_config(layout="wide")
st.title("🌿 Leaf Disease Detection and Severity Estimation")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Processing..."):
        input_image, output_image = remove_background(uploaded_file)

        # Predict type and disease
        leaf_type, disease = predict_leaf_and_disease(uploaded_file)
        st.subheader("🔍 Prediction Results")
        st.write(f"**Leaf Type**: {leaf_type}")
        st.write(f"**Disease**: {disease}")

        # Color ranges based on leaf type
        color_ranges = get_color_ranges(leaf_type)

        # Analyze disease severity
        green_mask, disease_mask, disease_percentage, features = analyze_leaf(output_image, color_ranges)

        st.subheader("📊 Disease Severity & Texture")
        st.write(f"**Diseased Area**: {disease_percentage:.2f}%")

        for feature, value in features.items():
            st.write(f"**{feature.capitalize()}**: {value:.4f}")

        st.subheader("🖼️ Visualization")
        col1, col2, col3, col4 = st.columns(4)
        col1.image(input_image, caption="Original", use_column_width=True)
        col2.image(output_image, caption="No Background", use_column_width=True)
        col3.image(green_mask, caption="Green Mask", use_column_width=True, channels="GRAY")
        col4.image(disease_mask, caption="Diseased Mask", use_column_width=True, channels="GRAY")
