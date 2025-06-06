import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

# Carregar e preparar os dados
df = pd.read_csv('ufabc-score-predictor/dados/bd_prograd01_2023_sem_identificadores.csv')
df['nota'] = df['nota'].astype(str).str.replace(',', '.').astype(float)

# Criar coluna binária: 1 para aprovado, 0 para não aprovado
status_aprovado = ['Matriculado(a)', 'Convidado(a) p/ Cadastro Reserva']
df['aprovado'] = df['status de matrícula'].isin(status_aprovado).astype(int)

# Features ampliadas
features = [
    'modalidade inscrição', 'opção de turno', 'opção de campus', 'nota',
    'classificação original', 'classificação atual', 'chamada', 'modalidade convocação'
]
target = 'aprovado'

# Tratar valores faltantes e tipos
for col in ['classificação original', 'classificação atual', 'chamada']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)
df['modalidade convocação'] = df['modalidade convocação'].fillna('Desconhecido')

# Separar X e y
X = df[features]
y = df[target]

# Pré-processamento (OneHot para categóricas)
categorical_cols = ['modalidade inscrição', 'opção de turno', 'opção de campus', 'modalidade convocação']
column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)
X_transformed = column_transformer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# GridSearch para melhores hiperparâmetros
param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_grid,
    cv=3,
    scoring='roc_auc',
    verbose=2,
    n_jobs=-1
)
grid.fit(X_train, y_train)

print("Melhores parâmetros:", grid.best_params_)

# Avaliação com melhores parâmetros
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
print(f"Acurácia: {acc:.2f} | ROC AUC: {roc:.2f}")