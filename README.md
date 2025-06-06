# 🐟 Fish Species Classifier with CNN

A deep learning-based web application to classify fish species from uploaded images using a Convolutional Neural Network (CNN). Built with **TensorFlow** and **Streamlit**.

---

## 🚀 Demo

![image](https://github.com/user-attachments/assets/389e78eb-18d1-44ff-b2c4-e25f138d0f7f)

> 🎯 Example Prediction: `fish sea_food trout`  
> ✅ Confidence: `92.5%`  
> 📊 Model Accuracy: `89.00%`

---

## 📌 Features

- 📸 Upload any fish image (.jpg/.jpeg/.png)
- 🧠 Classifies into 11 fish species
- ⚙️ Built using TensorFlow CNN model
- 🎨 Interactive web UI with Streamlit
- 📈 Displays prediction and model accuracy

---

## 🐍 Tech Stack

- Python 🐍
- TensorFlow / Keras
- NumPy
- Streamlit
- Matplotlib (for training visualization)

---

## 📁 Dataset

Used a categorized fish image dataset with subfolders for each species (train/val/test).  
> Format: `/train/fish sea_food trout/`, `/test/fish sea_food shrimp/`, etc.

---

## 🏗️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/fish-classifier-cnn.git
cd fish-classifier-cnn

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
