# 🧠 Productivity Intelligence System

### AI-Powered HR Analytics & Cognitive Load Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)]()
[![Machine Learning](https://img.shields.io/badge/ML-End%20to%20End-green.svg)]()
[![Neural Network](https://img.shields.io/badge/Neural%20Network-From%20Scratch-orange.svg)]()

---

## 🚀 Live Demo

🔗 **Streamlit App:**
https://appuctivity-intelligence-system-hkat3jcvvtq4esuwuffs5d.streamlit.app/

---

## 📌 Overview

The **Productivity Intelligence System** is an end-to-end machine learning application designed to predict employee productivity levels and cognitive fatigue risk using behavioral and environmental factors.

This project demonstrates the complete ML lifecycle:

* Synthetic data generation
* Data preprocessing pipelines
* Feature engineering
* Class imbalance handling
* Neural network implementation from scratch (NumPy)
* Model evaluation with custom metrics
* HR-style Streamlit dashboard deployment

The system simulates a real-world **HR analytics AI platform** for workforce performance monitoring.

---

## 🎯 Key Features

✅ Corporate HR-style dashboard UI
✅ Productivity level prediction (Low / Medium / High)
✅ Fatigue risk score estimation
✅ Probability visualization
✅ Behavioral analytics charts
✅ Neural network built from scratch
✅ Class imbalance handling (sampling + class weights)
✅ Mini-batch gradient descent
✅ Early stopping regularization
✅ Streamlit Cloud deployment

---

## 🧠 Machine Learning Pipeline

### 1️⃣ Synthetic Data Generation

Behavioral features simulated:

* Sleep hours
* Work hours
* Screen time
* Break frequency
* Task switching
* Stress level
* Hydration
* Noise level

---

### 2️⃣ Data Preprocessing

* Missing value handling
* Outlier clipping
* Feature engineering
* Scaling (StandardScaler)
* Label encoding
* Feature schema persistence

---

### 3️⃣ Feature Engineering

Derived features:

* Work-sleep ratio
* Stress per hour
* Break efficiency
* Focus index
* Screen fatigue

---

### 4️⃣ Class Imbalance Handling

Techniques used:

* Oversampling minority classes
* Class-weighted loss function

---

### 5️⃣ Neural Network (From Scratch)

Implemented using **NumPy only**:

Architecture:

Input → Hidden Layer → Output Layer (Softmax)

Forward propagation:

```
Z = W·X + b
A = ReLU(Z)
```

Softmax output:

```
ŷ = exp(z) / Σ exp(z)
```

Loss:

```
Cross-Entropy + L2 Regularization
```

Backpropagation:

```
∂L/∂W computed using chain rule
```

Optimization:

* Mini-batch gradient descent
* He initialization
* Dropout regularization
* Early stopping

---

### 6️⃣ Evaluation Metrics

Custom implementation:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1 Score

---

## 🏗️ Project Architecture

```
productivity-intelligence-system/
│
├── data/
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   └── feature_columns.pkl
│
├── src/
│   ├── preprocessing_pipeline.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── sampling.py
│   └── config.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 📊 Dashboard Capabilities

The Streamlit interface provides:

* Interactive employee behavior inputs
* Real-time productivity prediction
* Fatigue risk visualization
* Probability distribution charts
* Behavioral analytics overview

Designed to resemble an **enterprise HR analytics platform**.

---

## ⚙️ Installation (Local)

```bash
git clone https://github.com/Harshithpatali/productivity-intelligence-system.git
cd productivity-intelligence-system

pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deployment

The application is deployed using **Streamlit Cloud**.

Steps:

1. Push project to GitHub
2. Connect repository in Streamlit Cloud
3. Select `app.py`
4. Deploy

---

## 🧪 Model Performance

Example evaluation:

```
Accuracy: 99%

Class 0 → Precision: 0.92 | Recall: 1.00
Class 1 → Precision: 0.99 | Recall: 0.99
Class 2 → Precision: 1.00 | Recall: 0.99
```

---

## 🛠️ Tech Stack

* Python
* NumPy
* Pandas
* Scikit-learn
* Streamlit
* Matplotlib
* Joblib

---

## 💼 Resume Description

> Built an end-to-end HR analytics AI system predicting employee productivity and fatigue using a neural network implemented from scratch, advanced preprocessing pipelines, class imbalance handling, and deployed an interactive Streamlit dashboard.

---

## 🔮 Future Improvements

* SHAP explainability
* Docker containerization
* FastAPI backend
* Real HR dataset integration
* Model monitoring pipeline
* Kubernetes deployment

---

## 👨‍💻 Author

**Harshith Devraj**

* Machine Learning & AI Enthusiast
* Data Science | ML Engineering | Deep Learning

---

## ⭐ Acknowledgment

This project demonstrates practical ML engineering skills including model development, optimization, evaluation, and deployment in a production-style environment.

If you found this useful, consider giving the repository a ⭐.

---
