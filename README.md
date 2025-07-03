# Chronic Disease Prevention Tracker

This project builds an early-warning system to detect **chronic disease risks** (Diabetes, Obesity, High Blood Pressure) and recommend **preventive actions** based on user input and machine learning predictions.

---

## 📊 Dataset Source

- **Dataset**: [US Chronic Disease Indicators (CDI) 2023 – Kaggle](https://www.kaggle.com/datasets/payamamanat/us-chronic-disease-indicators-cdi-2023)
- The dataset includes public health data across states and years.
- Data was **cleaned** to remove missing values and filter only relevant health questions.
- Only **1% of the cleaned dataset** was used to build a lightweight yet functional prediction system.

---

## 🚀 Features

- 🧠 Early detection using Random Forest Classifier
- 📥 User input-driven predictions
- ✅ Validates input against real dataset values
- 🩺 Provides intervention strategies tailored to predicted risks
- 🧩 Modular architecture:
  - Monitoring engine
  - Prediction model
  - Intervention library
  - Risk tracking

---

## ⚙️ Workflow

1. **Data Preparation**
   - Filter on questions like:
     - "Diabetes among adults"
     - "Obesity among adults"
     - "High blood pressure"
   - Drop records with missing `DataValue`

2. **Risk Classification**
   - Convert numeric values into risk levels:
     - `< 7` → Low Risk
     - `7–14.99` → Moderate Risk
     - `≥ 15` → High Risk

3. **Model Training**
   - Features used:
     - `YearStart`, `LocationDesc`, `Stratification1`, `Question`
   - Model: `RandomForestClassifier` with preprocessing pipeline

4. **Prediction & Intervention**
   - Accept user input via terminal
   - Predict risk level
   - Suggest lifestyle interventions

---

## 📦 Requirements

```bash
pip install pandas scikit-learn
