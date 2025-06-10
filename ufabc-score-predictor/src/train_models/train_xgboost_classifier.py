import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Carregar e preparar os dados
df = pd.read_csv('C:/FaculProjects/hackaton-datas/ufabc-score-predictor/dados/bd_prograd01_2023_sem_identificadores.csv')
df['nota'] = df['nota'].astype(str).str.replace(',', '.').astype(float)

# Criar coluna binária: 1 para aprovado, 0 para não aprovado
# Ajuste os status conforme sua base
status_aprovado = ['Matriculado(a)']
df['aprovado'] = df['status de matrícula'].isin(status_aprovado).astype(int)

# Features e target
features = ['modalidade inscrição', 'opção de turno', 'opção de campus', 'opção de curso', 'nota']
target = 'aprovado'

# Separar X e y
X = df[features]
y = df[target]

# Pré-processamento (OneHot para categóricas)
categorical_cols = ['modalidade inscrição', 'opção de turno', 'opção de campus', 'opção de curso']
column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit só no treino, transform nos dois
X_train_transformed = column_transformer.fit_transform(X_train)
X_test_transformed = column_transformer.transform(X_test)

# Treinar modelo
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_transformed, y_train)

# Avaliação
y_pred = model.predict(X_test_transformed)
y_proba = model.predict_proba(X_test_transformed)[:, 1]
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
print(f"Acurácia: {acc:.2f} | ROC AUC: {roc:.2f}")

joblib.dump(model, 'C:\\FaculProjects\\hackaton-datas\\ufabc-score-predictor\\modelos_xgboost\\classifier_aprovacao.pkl')
joblib.dump(column_transformer, 'C:\\FaculProjects\\hackaton-datas\\ufabc-score-predictor\\modelos_xgboost\\classifier_transformer.pkl')

