## Project Description
This project predicts whether a customer is likely to churn using a machine learning model.
A Flask web application is built to collect user input and display churn prediction results.

## Technologies Used
- Python
- Flask
- XGBoost
- Scikit-learn
- Pandas
- HTML, CSS

## Machine Learning Model
- Algorithm used: XGBoost Classifier
- Input: Customer attributes
- Output: Churn / No Churn

## Project Structure
customer-churn-prediction/
│── app.py
│── README.md
│
├── model/
│   ├── xgboost_model.pkl
│   ├── model_columns.pkl
│   └── X_train_encoded.pkl
│
├── templates/
│   ├── form.html
│   └── result.html
│
├── static/

## How to Run
pip install -r requirements.txt  
python app.py  


## Output
- Displays customer churn prediction (Churn / No Churn)
- SHAP is used to explain global feature importance of the model
- LIME is used to provide local explanations for individual predictions
- Helps understand which features contribute most to customer churn


