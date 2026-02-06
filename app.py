from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
import joblib
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
import os
import uuid

app = Flask(__name__)

# Load model, columns, and training data
model = joblib.load("xgboost_model.pkl")
columns = joblib.load("model_columns.pkl")
X_train_encoded = joblib.load("X_train_encoded.pkl")

# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_encoded.values,
    feature_names=X_train_encoded.columns.tolist(),
    class_names=['No Churn', 'Churn'],
    mode='classification'
)

# Create SHAP explainer
shap_explainer = shap.Explainer(model, X_train_encoded)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Capture user input
    customer_id = request.form['customer_id']
    input_data = {
        'SeniorCitizen': int(request.form['SeniorCitizen']),
        'MonthlyCharges': float(request.form['MonthlyCharges']),
        'TotalCharges': float(request.form['TotalCharges']),
        'gender': request.form['gender'],
        'Partner': request.form['Partner'],
        'Dependents': request.form['Dependents'],
        'PhoneService': request.form['PhoneService'],
        'MultipleLines': request.form['MultipleLines'],
        'InternetService': request.form['InternetService'],
        'OnlineSecurity': request.form['OnlineSecurity'],
        'OnlineBackup': request.form['OnlineBackup'],
        'DeviceProtection': request.form['DeviceProtection'],
        'TechSupport': request.form['TechSupport'],
        'StreamingTV': request.form['StreamingTV'],
        'StreamingMovies': request.form['StreamingMovies'],
        'Contract': request.form['Contract'],
        'PaperlessBilling': request.form['PaperlessBilling'],
        'PaymentMethod': request.form['PaymentMethod'],
        'tenure': int(request.form['tenure'])
    }

    # 2. Preprocess input
    raw_df = pd.DataFrame([input_data])
    encoded = pd.get_dummies(raw_df)
    for col in columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[columns]

    # 3. Predict
    prediction = model.predict(encoded)[0]
    proba = model.predict_proba(encoded)[0]

    # 4. LIME explanation (save as image)
    lime_exp = explainer.explain_instance(encoded.values[0], model.predict_proba, num_features=10)
    img_id = str(uuid.uuid4())
    lime_img_filename = f"lime_explanation_{img_id}.png"
    lime_img_path = os.path.join("static", lime_img_filename)
    lime_fig = lime_exp.as_pyplot_figure()
    plt.tight_layout()
    lime_fig.savefig(lime_img_path)
    plt.close(lime_fig)

    # 5. SHAP explanation (save as image)
    shap_values = shap_explainer(encoded)
    shap_img_filename = f"shap_explanation_{img_id}.png"
    shap_img_path = os.path.join("static", shap_img_filename)
    shap.summary_plot(shap_values, encoded, show=False)
    plt.tight_layout()
    plt.savefig(shap_img_path)
    plt.close()

    # 6. Render result with download links
    return render_template('result.html',
                           customer_id=customer_id,
                           prediction="Churn" if prediction == 1 else "No Churn",
                           prob_churn=f"{proba[1]*100:.2f}%",
                           prob_no_churn=f"{proba[0]*100:.2f}%",
                           lime_explanation_image=lime_img_path,
                           lime_explanation_file=lime_img_filename,
                           shap_explanation_image=shap_img_path,
                           shap_explanation_file=shap_img_filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('static', filename, as_attachment=True)

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)
