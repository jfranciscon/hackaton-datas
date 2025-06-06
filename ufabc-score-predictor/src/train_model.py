import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

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

# --- Fluxo principal ---

# 1. Carregar os dados
df = load_data('dados/bd_prograd01_2023_sem_identificadores.csv')
df['nota'] = df['nota'].str.replace(',', '.').astype(float)

# 2. Pré-processar os dados
X, y, column_transformer = preprocess_data(df, target_col='nota', drop_cols=[])

# 3. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Instanciar e treinar o modelo
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 5. Fazer previsões e avaliar
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f} | R²: {r2:.2f}")

# Separar por modalidade
for modalidade, grupo in df.groupby('modalidade inscrição'):
    if len(grupo) < 2:
        print(f"Modalidade '{modalidade}' ignorada (menos de 2 amostras).")
        continue
    print(f"\nModalidade: {modalidade}")
    drop_cols = ['modalidade inscrição']
    X, y, column_transformer = preprocess_data(grupo, target_col='nota', drop_cols=drop_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE para {modalidade}: {mse:.2f} | R²: {r2:.2f}")