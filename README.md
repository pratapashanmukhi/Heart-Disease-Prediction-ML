# Heart Disease Prediction — Machine Learning Diagnostic Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A supervised Machine Learning classification study evaluating cardiovascular patient telemetry to detect coronary heart disease risks early, comparing Logistic Regression, Random Forest, K-Nearest Neighbors, and Support Vector Classifiers.

---

## Clinical Dataset Attributes (`heart.csv`)

| Attribute | Clinical Term | Description |
| :--- | :--- | :--- |
| **age** | Patient Age | Age in years |
| **sex** | Gender | 1 = male; 0 = female |
| **cp** | Chest Pain Type | 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic |
| **trestbps** | Resting Blood Pressure | mm Hg on admission to the hospital |
| **chol** | Serum Cholesterol | mg/dl |
| **fbs** | Fasting Blood Sugar | > 120 mg/dl (1 = true; 0 = false) |
| **restecg** | Resting ECG | 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy |
| **thalach** | Max Heart Rate | Maximum achieved heart rate during stress exercise |
| **exang** | Exercise Angina | 1 = yes; 0 = no |
| **oldpeak** | ST Depression | ST depression induced by exercise relative to rest |
| **slope** | ST Slope | Slope of the peak exercise ST segment |
| **ca** | Major Vessels | Number of major vessels (0–3) colored by flourosopy |
| **thal** | Thalassemia | 1 = normal; 2 = fixed defect; 3 = reversable defect |
| **target** | Diagnosis | 1 = Heart Disease Present; 0 = Heart Disease Absent |

---

## Machine Learning Pipeline

1. **Exploratory Data Analysis (EDA)**: Statistical distribution analysis, box plots for outlier detection, and correlation heatmap generation.
2. **Data Cleaning & Standardization**: Feature scaling via `StandardScaler` to prevent high-variance attributes (`chol`, `trestbps`) from skewing distance-based algorithms.
3. **Cross-Validation & Model Training**: 5-Fold stratified cross-validation testing multiple classifiers.
4. **Clinical Evaluation**: Metrics prioritized on **Sensitivity/Recall** to minimize dangerous false-negative clinical diagnoses.

---

## Project Structure

```
Heart-Disease-Prediction-ML/
├── Heart_Disease_Prediction_ML.ipynb   # Complete analysis and training notebook
├── heart.csv                           # Clinical patient dataset (303 records)
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Exclusion rules
└── README.md                           # Documentation
```

---

## Getting Started

```bash
git clone https://github.com/pratapashanmukhi/Heart-Disease-Prediction-ML.git
cd Heart-Disease-Prediction-ML
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter notebook Heart_Disease_Prediction_ML.ipynb
```

---

## Author

**Shanmukhi Pratapa**  
B.Tech CSE (AI & ML) — VIT-AP University  
GitHub: [@pratapashanmukhi](https://github.com/pratapashanmukhi)
