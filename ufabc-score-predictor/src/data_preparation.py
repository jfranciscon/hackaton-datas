import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Identify categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    # Define the ColumnTransformer
    column_transformer = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'  # Keep the rest of the columns unchanged
    )
    
    # Apply the transformations
    X = column_transformer.fit_transform(data.drop('target_column', axis=1))  # Replace 'target_column' with the actual target column name
    y = data['target_column']  # Replace 'target_column' with the actual target column name
    
    return X, y, column_transformer

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


