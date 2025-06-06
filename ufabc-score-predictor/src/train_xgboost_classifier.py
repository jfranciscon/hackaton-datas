import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

# Carregar e preparar os dados
df = pd.read_csv('ufabc-score-predictor/dados/bd_prograd01_2023_sem_identificadores.csv')
df['nota'] = df['nota'].astype(str).str.replace(',', '.').astype(float)

# Criar coluna binária: 1 para aprovado, 0 para não aprovado
# Ajuste os status conforme sua base
status_aprovado = ['Matriculado(a)', 'Convidado(a) p/ Cadastro Reserva']
df['aprovado'] = df['status de matrícula'].isin(status_aprovado).astype(int)

# Features e target
features = ['modalidade inscrição', 'opção de turno', 'opção de campus', 'nota']
target = 'aprovado'

# Separar X e y
X = df[features]
y = df[target]

# Pré-processamento (OneHot para categóricas)
categorical_cols = ['modalidade inscrição', 'opção de turno', 'opção de campus']
column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)
X_transformed = column_transformer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Treinar modelo
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
print(f"Acurácia: {acc:.2f} | ROC AUC: {roc:.2f}")

# Salvar modelo e transformer
joblib.dump(model, '../modelos_xgboost/classifier_aprovacao.pkl')
joblib.dump(column_transformer, '../modelos_xgboost/classifier_transformer.pkl')
print("Modelo e transformer salvos com sucesso!")