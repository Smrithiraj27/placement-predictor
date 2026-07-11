from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load all saved files
model = pickle.load(open('model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))
column_averages = pickle.load(open('column_averages.pkl', 'rb'))
explainer = pickle.load(open('explainer.pkl', 'rb'))

# Encoding maps — must match your LabelEncoder order exactly
ENCODINGS = {
    'gender': {'F': 0, 'M': 1},
    'ssc_b': {'Central': 0, 'Others': 1},
    'hsc_b': {'Central': 0, 'Others': 1},
    'hsc_s': {'Arts': 0, 'Commerce': 1, 'Science': 2},
    'degree_t': {'Comm&Mgmt': 0, 'Others': 1, 'Sci&Tech': 2},
    'workex': {'No': 0, 'Yes': 1},
    'specialisation': {'Mkt&Fin': 0, 'Mkt&HR': 1},
}

# Friendly names for SHAP explanation output
FEATURE_LABELS = {
    'gender': 'Gender',
    'ssc_p': '10th Grade %',
    'ssc_b': '10th Board',
    'hsc_p': '12th Grade %',
    'hsc_b': '12th Board',
    'hsc_s': '12th Stream',
    'degree_p': 'Degree %',
    'degree_t': 'Degree Type',
    'workex': 'Work Experience',
    'etest_p': 'Employability Test %',
    'specialisation': 'MBA Specialisation',
    'mba_p': 'MBA %',
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Build input row using form values, encoding categoricals.
    # Any blank/missing field silently falls back to the column average.
    input_data = {}
    for col in columns:
        raw_value = request.form.get(col)

        if col in ENCODINGS:
            if raw_value:
                input_data[col] = ENCODINGS[col].get(raw_value, column_averages[col])
            else:
                input_data[col] = column_averages[col]
        else:
            if raw_value:
                try:
                    input_data[col] = float(raw_value)
                except ValueError:
                    input_data[col] = column_averages[col]
            else:
                input_data[col] = column_averages[col]

    input_df = pd.DataFrame([input_data])[columns]

    # Prediction + confidence
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    confidence = round(max(probability) * 100, 1)

    result = "Placed" if prediction == 1 else "Not Placed"

    # SHAP explanation — top 3 contributing features
    shap_values = explainer(input_df)
    values = shap_values.values[0]

    contributions = list(zip(columns, values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_3 = contributions[:3]

    explanation = []
    for feature, value in top_3:
        direction = "increased" if value > 0 else "decreased"
        explanation.append(f"{FEATURE_LABELS[feature]} {direction} placement likelihood")

    return render_template(
        'index.html',
        prediction_text=result,
        confidence=confidence,
        explanation=explanation
    )


if __name__ == '__main__':
    app.run(debug=True)
