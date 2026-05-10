# 🎓 Student Dropout Risk Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-dropout-risk-predictor-main.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)

A professional-grade Machine Learning application designed to predict the likelihood of a student dropping out of higher education. Developed as part of the **Applied Machine Learning - EDGE Series** at **Dhaka University of Engineering & Technology (DUET), Gazipur**.

## 📝 Project Overview

This tool helps educational institutions identify at-risk students early by analyzing demographic, socioeconomic, and academic performance data. By providing early warnings, advisors can intervene proactively to support students and improve retention rates.

### Key Features

- **Real-time Prediction**: Instantly calculate dropout probability based on 30+ input features.
- **Dynamic Risk Levels**: Visual feedback using Green (Low), Yellow (Medium), and Red (High) risk indicators.
- **Gauge Visualization**: Interactive Plotly gauge charts showing the risk score.
- **Top Risk Factors**: Automated identification of the top 3 specific factors contributing to a student's risk level.
- **Feature Importance Analysis**: Interactive bar charts showing which data points impact the model most.
- **Model Performance Dashboard**: Detailed transparency into the model's accuracy, F1-score, and AUC-ROC metrics.
- **Actionable Recommendations**: Specific intervention steps based on the calculated risk level.

## 🔗 Live Demo

You can access the live application here:
**[https://student-dropout-risk-predictor-main.streamlit.app](https://student-dropout-risk-predictor-main.streamlit.app)**

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/naimur-naiyimu/Student_Dropout_Risk_Predictor.git
   cd Student_Dropout_Risk_Predictor
   ```
2. **Create a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application**:

   ```bash
   streamlit run app.py
   ```

## 📊 Dataset Information

The model is trained on the **UCI Student Dropout and Academic Success dataset**, which includes:

- **Demographic data**: Marital status, nationality, gender, etc.
- **Socioeconomic data**: Parent's qualification and occupation, scholarship status.
- **Academic data**: Performance in 1st and 2nd semesters (credits, grades, evaluations).
- **Macroeconomic data**: Unemployment rate, inflation, GDP.

## 🧠 Model Details

- **Algorithm**: Balanced Ensemble/Optimized Classifier (e.g., Random Forest or XGBoost).
- **Optimization**: Hyperparameter tuning via GridSearchCV.
- **Class Balancing**: SMOTE (Synthetic Minority Over-sampling Technique) applied to handle imbalanced dropout data.
- **Evaluation**: 5-fold Stratified Cross-Validation.

## 🔮 Future Roadmap

- [ ] **Batch Processing**: Allow users to upload CSV files for bulk student risk analysis.
- [ ] **API Access**: Create a REST API for integration with existing University Management Systems (UMS).
- [ ] **Email Alerts**: Automated email notifications to advisors when a student hits "High Risk".
- [ ] **Historical Tracking**: Store prediction history to see how a student's risk changes over semesters.
- [ ] **Localized Support**: Localization for different educational systems and grading scales.

## 👨‍💻 Developer

* **Naimur Rahman**

---

**Instructors:**

- Prof. Dr. Fazlul Hasan Siddiqui
- Md. Rahad Khan

**Project Series:** Applied Machine Learning - EDGE Series, DUET, Gazipur.
