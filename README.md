# Placement Predictor

A machine learning web app that predicts a student's likelihood of campus placement based on academic record, with model interpretability powered by SHAP.

## Overview

This project predicts whether a student is likely to be placed during campus recruitment, using academic and demographic features (10th/12th grade percentages, degree performance, work experience, and more). Beyond a simple yes/no output, it returns a **confidence score** and the **top contributing factors** behind each prediction, using SHAP (SHapley Additive exPlanations).

The dataset used is Ben Roshan's [Campus Recruitment dataset](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement) on Kaggle, which covers MBA campus placements — so this model's scope is specific to that context rather than general engineering placements.

## Why this project

I wanted to apply the same interpretability approach I used in my published research on cancer prognosis (SHAP + ensemble methods + survival analysis) to a smaller, self-contained classification problem — treating model transparency as a core requirement, not an afterthought.

## Model selection

Two models were trained and compared on precision, recall, and F1-score, not just raw accuracy:

| Model | Accuracy | Precision (Placed) | Recall (Placed) | F1 (Placed) |
|---|---|---|---|---|
| Logistic Regression | **88%** | 0.91 | 0.94 | 0.92 |
| Random Forest | 79% | 0.82 | 0.90 | 0.86 |

**Logistic Regression was selected for deployment.** On this dataset size (~215 rows), Random Forest showed signs of overfitting, while Logistic Regression generalized better across both classes. This was a deliberate, evidence-based choice rather than defaulting to the more complex model.

## Features used

| Feature | Description |
|---|---|
| `gender` | Student gender |
| `ssc_p` | 10th grade percentage |
| `ssc_b` | 10th board (Central / Others) |
| `hsc_p` | 12th grade percentage |
| `hsc_b` | 12th board |
| `hsc_s` | 12th stream (Science / Commerce / Arts) |
| `degree_p` | Undergraduate degree percentage |
| `degree_t` | Degree type |
| `workex` | Prior work experience |
| `etest_p` | Employability test score |
| `specialisation` | MBA specialization |
| `mba_p` | MBA percentage |

`etest_p` and `mba_p` are optional in the app — if left blank, they default to the dataset average rather than forcing an artificial value.

## How it works

1. **Data preprocessing** — nulls handled, categorical fields label-encoded, irrelevant columns (serial number, salary) dropped.
2. **Model training** — Logistic Regression and Random Forest trained and compared using `scikit-learn`.
3. **Explainability** — a SHAP explainer is fit on the training data to attribute each prediction to its top contributing features.
4. **Serving** — a Flask backend loads the trained model, encoders, and SHAP explainer, and serves predictions through a simple web form.

## Tech stack

- **Python** — data processing and modeling
- **pandas** — data cleaning and manipulation
- **scikit-learn** — model training and evaluation
- **SHAP** — model interpretability
- **Flask** — web application backend
- **HTML/CSS** — frontend

## Running locally

```bash
git clone https://github.com/YOUR_USERNAME/placement-predictor.git
cd placement-predictor
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Project structure

```
placement-predictor/
├── app.py                  # Flask application
├── model.pkl                # Trained Logistic Regression model
├── columns.pkl               # Feature column order
├── column_averages.pkl        # Default values for optional fields
├── explainer.pkl              # SHAP explainer
├── requirements.txt
└── templates/
    └── index.html            # Frontend form + results view
```

## Future improvements

- Feature importance visualization on the results page
- Support for a general (non-MBA) placement dataset
- Model retraining pipeline with cross-validation

## Author

Smrithi Raj — B.Tech Computer Science Engineering, Guru Nanak Institute of Technology
