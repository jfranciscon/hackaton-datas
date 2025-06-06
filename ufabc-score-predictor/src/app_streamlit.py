import streamlit as st
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Carregar dados
df = pd.read_csv('C:\\FaculProjects\\hackaton-datas\\ufabc-score-predictor\\dados\\bd_prograd01_2023_sem_identificadores.csv')
df['nota'] = df['nota'].astype(str).str.replace(',', '.').astype(float)

group_cols = ['modalidade inscrição', 'opção de turno', 'opção de campus']
model_dict = {}

# Treinar um modelo de regressão para cada grupo
for group_keys, grupo in df.groupby(group_cols):
    if len(grupo) < 2:
        continue
    X = pd.DataFrame([[0]] * len(grupo))  # Dummy feature
    y = grupo['nota']
    model = XGBRegressor()
    model.fit(X, y)
    model_dict[group_keys] = model

# --- Treinar classificador na hora ---
status_aprovado = ['Matriculado(a)', 'Convidado(a) p/ Cadastro Reserva']
df['aprovado'] = df['status de matrícula'].isin(status_aprovado).astype(int)

features = [
    'modalidade inscrição', 'opção de turno', 'opção de campus', 'nota'
]
categorical_cols = ['modalidade inscrição', 'opção de turno', 'opção de campus']

X_cls = df[features]
y_cls = df['aprovado']

column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)
X_cls_transformed = column_transformer.fit_transform(X_cls)

classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
classifier.fit(X_cls_transformed, y_cls)

# Carregar menores notas por grupo, considerando apenas "Matriculado(a)"
df_matriculado = df[df['status de matrícula'] == 'Matriculado(a)']
df_menor = (
    df_matriculado.groupby(group_cols)['nota']
    .min()
    .reset_index()
    .set_index(group_cols)
)

# Interface Streamlit
st.title("Previsão de Nota e Probabilidade de Aprovação")

modalidades = df['modalidade inscrição'].unique().tolist()
turnos = df['opção de turno'].unique().tolist()
campi = df['opção de campus'].unique().tolist()

modalidade = st.selectbox('Modalidade de Inscrição', modalidades)
turno = st.selectbox('Opção de Turno', turnos)
campus = st.selectbox('Opção de Campus', campi)

group_key = (modalidade, turno, campus)

if st.button("Prever Nota e Probabilidade"):
    if group_key in model_dict:
        X_input = pd.DataFrame([[0]])  # Dummy feature
        nota_prevista = model_dict[group_key].predict(X_input)[0]
        st.success(f"Nota prevista: {nota_prevista:.2f}")

        # Preparar entrada para o classificador
        input_dict = {
            'modalidade inscrição': modalidade,
            'opção de turno': turno,
            'opção de campus': campus,
            'nota': nota_prevista
        }
        X_input_cls = pd.DataFrame([input_dict])
        X_input_cls_transformed = column_transformer.transform(X_input_cls)
        prob_aprov = classifier.predict_proba(X_input_cls_transformed)[0, 1]
        st.info(f"Probabilidade de aprovação: {prob_aprov*100:.1f}%")

        # Mostrar a menor nota do grupo considerando apenas "Matriculado(a)"
        if group_key in df_menor.index:
            menor_nota = df_menor.loc[group_key]['nota']
            st.write(f"Menor nota de matriculado(a) nesse grupo: {menor_nota:.2f}")
            if nota_prevista >= menor_nota:
                st.success("Sua nota prevista está acima ou igual ao menor matriculado deste grupo!")
            else:
                st.warning("Sua nota prevista está abaixo do menor matriculado deste grupo.")
        else:
            st.write("Não há referência de menor nota de matriculado(a) para esse grupo.")
    else:
        st.warning("Não há modelo treinado para essa combinação de modalidade, turno e campus.")