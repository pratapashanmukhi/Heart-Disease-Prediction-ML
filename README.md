# Heart Disease Prediction - Supervised Clinical Diagnostic Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A supervised clinical Machine Learning diagnostic study evaluating patient cardiovascular telemetry to detect coronary heart disease risks early. Evaluates feature correlations across 13 physiological biomarkers and compares Logistic Regression, Decision Trees, and Random Forest ensembles.

---

## Project Overview

Cardiovascular diseases (CVDs) remain the leading cause of global mortality. Early identification of cardiac pathologies allows medical professionals to initiate preventative lifestyle interventions and targeted therapies prior to irreversible myocardial infarction.

This study implements an end-to-end clinical machine learning diagnostic pipeline:
- **Biomarker Exploratory Data Analysis**: Analyzes physiological metrics across 1,025 cardiac patient telemetry records (`heart.csv`).
- **Feature Correlation & Preprocessing**: Evaluates clinical attributes including resting electrocardiographic anomalies, chest pain classifications, exercise-induced angina, and fluoroscopy vessel counts.
- **Model Evaluation & Clinical Prioritization**: Trains and compares linear baselines against non-linear tree ensembles, achieving up to **98.54% classification accuracy**.

### Problem Statement
- **Diagnostic Latency**: Comprehensive angiographic catheterization and invasive stress testing require specialized cardiac laboratories and carry procedural risks.
- **Asymptomatic Pathology**: Patients frequently exhibit silent ischemia where standard resting symptoms fail to signal underlying coronary artery blockages.
- **False Negative Severity**: In clinical triage, false negatives (failing to diagnose an at-risk cardiac patient) carry catastrophic life-threatening outcomes compared to false alarms.

---

## Model Benchmark Results

Evaluated on the 1,025-record cardiovascular patient dataset (80/20 train/test split):

| Machine Learning Classifier | Training Strategy | Test Accuracy | Clinical Suitability |
| :--- | :--- | :--- | :--- |
| **Random Forest Classifier** | **100 Decision Trees (random_state=42)** | **98.54%** | **Optimal - High sensitivity and robustness against feature variance** |
| **Decision Tree Classifier** | Standard CART Algorithm | **98.54%** | Strong - High interpretability with clear diagnostic split criteria |
| **Logistic Regression** | L-BFGS Solver (`max_iter=1000`) | **79.51%** | Baseline - Limited by non-linear biomarker interactions |

---

## Clinical Biomarkers (`heart.csv`)

| Biomarker | Clinical Parameter | Values / Units | Diagnostic Significance |
| :--- | :--- | :--- | :--- |
| **age** | Patient Age | Years | Non-modifiable cardiovascular risk factor |
| **sex** | Gender | 1 = male; 0 = female | Demographic baseline stratification |
| **cp** | Chest Pain Type | 0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic | Primary symptomatic indicator |
| **trestbps** | Resting Blood Pressure | mm Hg on admission | Systemic arterial hypertension indicator |
| **chol** | Serum Cholesterol | mg/dl | Hyperlipidemia and plaque accumulation metric |
| **fbs** | Fasting Blood Sugar | 1: > 120 mg/dl; 0: normal | Diabetes comorbidity marker |
| **restecg** | Resting Electrocardiogram | 0: Normal, 1: ST-T wave, 2: Ventricular hypertrophy | Conduction and ventricular abnormalities |
| **thalach** | Max Heart Rate Achieved | Beats per minute (bpm) | Cardiovascular functional reserve |
| **exang** | Exercise-Induced Angina | 1 = yes; 0 = no | Stress-induced ischemic symptoms |
| **oldpeak** | ST Depression | Numeric (mm) | Exercise-induced myocardial ischemia |
| **slope** | Peak ST Segment Slope | 0: Upsloping, 1: Flat, 2: Downsloping | Ischemic severity during maximal stress |
| **ca** | Major Fluoroscopy Vessels | 0 to 3 vessels | Direct radiological vessel patency count |
| **thal** | Thalassemia Stress Scan | 0: Unknown, 1: Normal, 2: Fixed defect, 3: Reversible | Myocardial perfusion defect permanence |
| **target** | Cardiac Pathology Status | **1 = Diseased; 0 = Normal** | Ground truth clinical diagnosis |

---

## Machine Learning Pipeline

```
[ Clinical Patient Telemetry (heart.csv - 1,025 records) ]
                            │
                            ▼
     [ Exploratory Data Analysis & Quality Validation ]
      ├── Missing Value & Outlier Verification
      ├── Feature Distribution & Correlation Heatmaps
      └── Target Stratification (526 Diseased / 499 Normal)
                            │
                            ▼
         [ Feature Preprocessing & Partitioning ]
      ├── Train/Test Split (80% Training / 20% Holdout Testing)
      └── Numerical Attribute Normalization
                            │
                            ▼
          [ Multi-Model Training & Benchmarking ]
      ├── Logistic Regression (Baseline: 79.51%)
      ├── Decision Tree Classifier (98.54%)
      └── Random Forest Classifier Ensemble (98.54%)
                            │
                            ▼
           [ Clinical Evaluation & Diagnostic Insights ]
      └── Metric Prioritization & Decision Support Interpretation
```

---

## Project Structure

```
Heart-Disease-Prediction-ML/
├── Heart_Disease_Prediction_ML.ipynb   # Complete analysis, training, and evaluation notebook
├── heart.csv                           # Clinical patient dataset (1,025 records)
├── requirements.txt                    # Pinned Python dependencies
├── .gitignore                          # Checkpoints, bytecode, and cache exclusions
└── README.md                           # Comprehensive documentation
```

---

## Getting Started

### Prerequisites

- Python 3.9, 3.10, or 3.11
- pip package manager

### Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pratapashanmukhi/Heart-Disease-Prediction-ML.git
   cd Heart-Disease-Prediction-ML
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows:
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook Heart_Disease_Prediction_ML.ipynb
   ```

---

## Author

**Shanmukhi Pratapa**  
B.Tech Computer Science & Engineering (Artificial Intelligence & Machine Learning)  
VIT-AP University  
GitHub: [@pratapashanmukhi](https://github.com/pratapashanmukhi)
