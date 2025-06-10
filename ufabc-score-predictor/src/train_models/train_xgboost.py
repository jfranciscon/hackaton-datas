import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import os
from unidecode import unidecode  # Importante!

def preprocess_data(data, target_col, drop_cols):
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    for col in [target_col] + drop_cols:
        if col in categorical_cols:
            categorical_cols.remove(col)
    column_transformer = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )
    X = column_transformer.fit_transform(data.drop([target_col] + drop_cols, axis=1))
    y = data[target_col]
    return X, y, column_transformer

# Caminho dos dados
df = pd.read_csv('C:/FaculProjects/hackaton-datas/ufabc-score-predictor/dados/bd_prograd01_2023_sem_identificadores.csv')
df['nota'] = df['nota'].astype(str).str.replace(',', '.').astype(float)

# Agrupar por curso
group_cols = ['opção de curso']

# Diretório de saída
save_dir = 'C:/FaculProjects/hackaton-datas/ufabc-score-predictor/modelos_xgboost/regressores_por_curso'
os.makedirs(save_dir, exist_ok=True)

for group_keys, grupo in df.groupby(group_cols):
    if len(grupo) < 2:
        print(f"Grupo {group_keys} ignorado (menos de 2 amostras).")
        continue

    print(f"\nGrupo: {group_keys}")
    drop_cols = group_cols
    X, y, column_transformer = preprocess_data(grupo, target_col='nota', drop_cols=drop_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.2f} | R²: {r2:.2f}")

    # Normalize o nome do curso com unidecode
    curso_nome = unidecode(str(group_keys[0])).replace('/', '_').replace('\\', '_').replace(' ', '_').upper()

    # Salvar modelo e transformer
    joblib.dump(model, f"{save_dir}/regressor_{curso_nome}_.pkl")
    joblib.dump(column_transformer, f"{save_dir}/transformer_{curso_nome}_.pkl")
