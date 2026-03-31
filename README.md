# Resume Intelligence System: ML-Based Resume Analysis and Role Prediction

An end-to-end machine learning system for automated resume screening, role prediction, and candidate readiness scoring using structured feature engineering, calibrated classification, and explainable skill diagnostics.

---

## Problem Statement

Recruiters handling large resume volumes face:
- Time-consuming manual screening
- Inconsistent role matching
- Limited objective scoring
- Poor visibility into candidate skill gaps

This project addresses those challenges with an automated, data-driven resume analysis pipeline.

---

## Solution Overview

The system processes uploaded resumes and applies ML models to:
- Predict the most suitable job role
- Generate a readiness score on a 0-100 scale
- Detect skill gaps against role profiles
- Provide explainability using probabilities, coverage checks, and role diagnostics

---

## System Architecture

~~~text
Resume PDF -> Text Parsing -> Feature Extraction -> ML Inference
             -> Readiness Regression
             -> Role Classification (calibrated probabilities)
             -> Skill Coverage Guardrails
             -> Dashboard Insights
~~~

---

## Key Features

- Automated PDF resume parsing and skill extraction
- Job role prediction using classification models
- Readiness scoring on a 0-100 scale using regression models
- Skill gap analysis against role-specific requirements
- Explainability layer with model confidence and threshold checks
- Interactive Streamlit dashboard with:
- Home market dashboard
- Resume analyzer
- Model comparison view

---

## Machine Learning Approach

### Classification (Job Role Prediction)
- Random Forest (best performer)
- Logistic Regression
- SVM
- KNN

### Regression (Readiness Score)
- Gradient Boosting (best performer)
- Random Forest
- Ridge Regression
- SVR
- KNN Regressor

---

## Verified Model Performance

- Best classification model: Random Forest
- Test accuracy: 72.71%
- Test top-3 accuracy: 99.17%
- Test weighted precision: 73.44%
- Test weighted F1 score: 72.17%

- Best regression model: Gradient Boosting
- Test MAE: 0.0078
- Test R2: 1.0000

Dataset summary:
- Total resumes: 3,200
- Job roles: 16
- Engineered features: 15
- Split sizes: train 2,239, validation 481, test 480

---

## Feature Engineering

The model uses structured numeric features including:
- Skill count
- Project count
- Certification count
- Internship flag
- Skill diversity
- Domain indicators:
- Python
- ML
- Web
- Cloud
- Database
- NLP
- DevOps
- Mobile
- Security
- Design

These features are transformed into fixed numeric vectors for training and inference.

---

## Tech Stack

- Frontend and dashboard: Streamlit
- Backend and processing: Python
- Core libraries:
- pandas
- numpy
- scikit-learn
- plotly
- pdfplumber
- joblib

---

## Project Structure

~~~text
resume_app/
  app.py
  train_models.py
  requirements.txt
  data/
    structured_resumes.csv
    generate_cse_dataset.py
  models/
    classification_model.pkl
    regression_model.pkl
    label_encoder.pkl
    feature_names.pkl
    model_comparison.json
    thresholds_config.json
  utils/
    parser.py
    features.py
    role_recommender.py
    model_loader.py
  views/
    home.py
    analyzer.py
    model_comparison.py
~~~

---

## Pipeline Flow

1. Upload a text-based PDF resume
2. Parse raw text and extract skill signals
3. Generate engineered numeric features
4. Run:
- Classification for role prediction
- Regression for readiness score
5. Apply role-coverage diagnostics and thresholds
6. Display:
- Final role recommendation
- Readiness score
- Top predictions
- Skill gaps and explainability metrics

---

## Limitations

- Trained on synthetic resume data, so real-world variation is underrepresented
- Performance depends heavily on feature engineering quality
- Generalization can improve with larger real-world datasets and external validation

---

## Future Improvements

- Incorporate real-world resume datasets
- Add job description to resume matching
- Expand NLP with transformer-based models
- Expose inference as an API service
- Integrate with recruiting platforms and ATS workflows

---

## Setup and Run

~~~bash
cd resume_app
pip install -r requirements.txt
python train_models.py
streamlit run app.py
~~~

---

## Key Takeaways

This project demonstrates:
- End-to-end ML pipeline design
- Practical feature engineering and evaluation
- Explainable prediction workflows
- A usable dashboard for hiring-oriented decision support
