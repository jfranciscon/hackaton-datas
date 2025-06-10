import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from numpy import exp

# Caminho dos dados
DATA_PATH = 'C:/FaculProjects/hackaton-datas/ufabc-score-predictor/dados/bd_prograd01_2023_sem_identificadores.csv'

# Carregar dados
df = pd.read_csv(DATA_PATH)
df['nota'] = df['nota'].astype(str).str.replace(',', '.').astype(float)
df['status de matrícula'] = df['status de matrícula'].str.strip().str.lower()

# Remover outliers (IQR)
Q1 = df['nota'].quantile(0.25)
Q3 = df['nota'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['nota'] >= Q1 - 1.5 * IQR) & (df['nota'] <= Q3 + 1.5 * IQR)]

# Features para regressão
features = ['modalidade inscrição', 'opção de turno', 'opção de campus', 'opção de curso']
target = 'nota'
categorical_cols = features

# Pré-processamento
column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)
X = column_transformer.fit_transform(df[features])
y = df[target]

# Treinar modelo de regressão geral
regressor = XGBRegressor()
regressor.fit(X, y)

# Calcular média da nota dos convocados por grupo
group_cols = features
df_convocado = df[df['status de matrícula'] == 'matriculado(a)']
media_notas_por_grupo = (
    df_convocado
    .groupby(group_cols)['nota']
    .mean()
    .reset_index()
    .rename(columns={'nota': 'nota_media_convocados'})
)

# Função para probabilidade suavizada (sigmoide)
def calcular_probabilidade_suavizada(nota_prevista, nota_media):
    k = 0.08  # controle da inclinação da curva
    x = nota_prevista - nota_media
    prob = 1 / (1 + exp(-k * x))
    return 15 + prob * 70  # escala para [15, 85]

# Interface Streamlit
st.title("Previsão de Nota e Probabilidade de Aprovação")

modalidades = df['modalidade inscrição'].unique().tolist()
turnos = df['opção de turno'].unique().tolist()
campi = df['opção de campus'].unique().tolist()
cursos = df['opção de curso'].unique().tolist()

modalidade = st.selectbox('Modalidade de Inscrição', modalidades)
turno = st.selectbox('Opção de Turno', turnos)
campus = st.selectbox('Opção de Campus', campi)
curso = st.selectbox('Opção de Curso', cursos)

if st.button("Prever Nota e Probabilidade"):
    # Montar entrada para regressão
    input_dict = {
        'modalidade inscrição': modalidade,
        'opção de turno': turno,
        'opção de campus': campus,
        'opção de curso': curso
    }
    X_input = pd.DataFrame([input_dict])
    X_input_transformed = column_transformer.transform(X_input)
    nota_prevista = regressor.predict(X_input_transformed)[0]
    st.success(f"Nota média prevista para este grupo: {nota_prevista:.2f}")

    # Calcular média do grupo
    filtro = (
        (media_notas_por_grupo['modalidade inscrição'] == modalidade) &
        (media_notas_por_grupo['opção de turno'] == turno) &
        (media_notas_por_grupo['opção de campus'] == campus) &
        (media_notas_por_grupo['opção de curso'] == curso)
    )
    media_valor = media_notas_por_grupo.loc[filtro, 'nota_media_convocados'].values[0] if not media_notas_por_grupo.loc[filtro].empty else 650.0

    # Calcular probabilidade suavizada
    prob_percent = calcular_probabilidade_suavizada(nota_prevista, media_valor)

    # Exibir barra de progresso
    st.write("### Probabilidade de aprovação:")
    st.progress(int(prob_percent))
    st.info(f"{prob_percent:.1f}% de chance")

    # Feedback motivacional
    if prob_percent >= 80:
        st.success("Alta chance de aprovação! 🚀")
    elif prob_percent >= 50:
        st.warning("Chance moderada. Continue focado nos estudos! 🔄")
    else:
        st.error("Chance mais baixa. Mas você pode melhorar com esforço! 📚")

    # Comparar com a média do grupo
    st.write(f"Nota média dos convocados neste grupo: **{media_valor:.2f}**")
    if nota_prevista >= media_valor:
        st.success("Sua nota está acima da média de convocados do grupo. Você está no caminho certo!")
    else:
        st.warning("Sua nota ainda está abaixo da média de convocados. Continue se dedicando!")
