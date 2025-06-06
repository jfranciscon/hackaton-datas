from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_model(model_path):
    return joblib.load(model_path)

def preprocess_input(input_data, categorical_features, numerical_features):
    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data, index=[0])
    
    # Preprocess categorical features
    encoder = OneHotEncoder(handle_unknown='ignore')
    input_encoded = encoder.fit_transform(input_df[categorical_features]).toarray()
    
    # Combine encoded categorical features with numerical features
    input_numerical = input_df[numerical_features].values
    processed_input = np.concatenate([input_encoded, input_numerical], axis=1)
    
    return processed_input

def make_prediction(model, processed_input):
    return model.predict(processed_input)

if __name__ == "__main__":
    model_path = 'path/to/your/trained_model.pkl'  # Update with your model path
    categorical_features = ['feature1', 'feature2']  # Update with your categorical features
    numerical_features = ['feature3', 'feature4']  # Update with your numerical features
    
    # Example input data
    input_data = {
        'feature1': 'value1',
        'feature2': 'value2',
        'feature3': 10,
        'feature4': 20
    }
    
    model = load_model(model_path)
    processed_input = preprocess_input(input_data, categorical_features, numerical_features)
    prediction = make_prediction(model, processed_input)
    
    print(f'Predicted minimum score required for approval: {prediction[0]}')