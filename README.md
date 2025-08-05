# 🌿 Leaf Disease Detection and Severity Estimation

This project is a deep learning-powered web application built using **Streamlit** that can:

- Detect the **type of leaf** (Apple, Cotton, Grape, Mango).
- Identify the **specific disease** affecting the leaf.
- Estimate the **severity of the disease** as a percentage.
- Visualize the **healthy vs diseased** regions.
- Compute **texture features** using GLCM (contrast, homogeneity, energy, correlation).

---

## 🚀 Features

- 📸 Upload your own leaf image.
- 🤖 Uses **CNN models** (Keras `.h5` format) for both leaf and disease classification.
- 🧠 Leverages **rembg** to remove background for better analysis.
- 🎨 Detects green and brown color regions to calculate diseased area.
- 📈 Extracts **GLCM-based texture features**.
- 📊 Interactive Streamlit dashboard with side-by-side visualization.

---

## 🧠 Tech Stack

| Technology | Purpose |
|------------|---------|
| `Streamlit` | Frontend interface |
| `Keras (TensorFlow)` | Deep learning models |
| `OpenCV` | Image preprocessing |
| `rembg` | Background removal |
| `skimage` | GLCM feature extraction |
| `matplotlib` | Visualizations |
| `Pillow` | Image handling |

---

## 📁 Project Structure

```bash
leaf-disease-app/
├── app.py                     # Main Streamlit app
├── leaf_type.h5               # Leaf type classification model
├── model_leaf_apple.h5        # Apple leaf disease model
├── model_leaf_mango.h5        # Mango leaf disease model
├── model_leaf_grape.h5        # Grape leaf disease model
├── model_leaf_cotton.h5       # Cotton leaf disease model
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
🧪 How It Works
Upload a leaf image (.jpg, .png, .jpeg).

The app first uses a leaf classification model to detect the type.

Then, a disease model specific to the leaf type is loaded and used to classify the disease.

The background is removed using rembg, and the disease area is estimated using color masking.

GLCM texture features are extracted from the processed image.

📦 Installation
bash
Copy
Edit
# Clone the repository
git clone https://github.com/your-username/leaf-disease-app.git
cd leaf-disease-app

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
🔒 Note
This repository contains only the structure and logic. The actual .h5 model files are private and not included in the public repo for licensing and security reasons. You may contact the author if you wish to access the full codebase or models.

👤 Author
Avighyan Chakraborty
📫 LinkedIn
📧 your-email@example.com
