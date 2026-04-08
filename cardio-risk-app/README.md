# Cardiovascular Risk Screening Tool

## Overview
This project builds a machine learning-based cardiovascular risk screening system using NHANES data.

The system provides:
- User-input-based risk prediction (non-clinical)
- Full-variable model with laboratory features
- Risk stratification instead of diagnosis

## Features
- Logistic Regression (user model)
- HistGradientBoosting (full model)
- Probability calibration (Platt scaling)
- Threshold optimization (precision-constrained)
- Streamlit interactive UI

## Model Performance
- ROC AUC: ~0.87 (user model)
- Recall prioritized screening design
- Designed for low prevalence (~4–5%)

## Disclaimer
This tool is for screening purposes only and does not constitute medical advice.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app/app.py