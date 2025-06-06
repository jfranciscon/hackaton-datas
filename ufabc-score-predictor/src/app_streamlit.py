import streamlit as st
import pandas as pd
from xgboost import XGBRegressor

# Carregar dados
df = pd.read_csv('C:\\FaculProjects\\hackaton-datas\\ufabc-score-predictor\\dados\\bd_prograd01_2023_sem_identificadores.csv')
df['nota'] = df['nota'].astype(str).str.replace(',', '.').astype(float)

group_cols = ['modalidade inscrição', 'opção de turno', 'opção de campus']
model_dict = {}

# Treinar um modelo para cada grupo
for group_keys, grupo in df.groupby(group_cols):
    if len(grupo) < 2:
        continue
    X = pd.DataFrame([[0]] * len(grupo))  # Dummy feature, pois não há outras features
    y = grupo['nota']
    model = XGBRegressor()
    model.fit(X, y)
    model_dict[group_keys] = model

# Interface Streamlit
st.title("Previsão de Nota por Modalidade, Turno e Campus")

modalidades = df['modalidade inscrição'].unique().tolist()
turnos = df['opção de turno'].unique().tolist()
campi = df['opção de campus'].unique().tolist()

modalidade = st.selectbox('Modalidade de Inscrição', modalidades)
turno = st.selectbox('Opção de Turno', turnos)
campus = st.selectbox('Opção de Campus', campi)

group_key = (modalidade, turno, campus)

if st.button("Prever Nota"):
    if group_key in model_dict:
        X_input = pd.DataFrame([[0]])  # Dummy feature
        nota_prevista = model_dict[group_key].predict(X_input)[0]
        st.success(f"Nota prevista: {nota_prevista:.2f}")
    else:
        st.warning("Não há modelo treinado para essa combinação de modalidade, turno e campus.")