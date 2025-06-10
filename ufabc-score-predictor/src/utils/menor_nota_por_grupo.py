import pandas as pd

# Carregar os dados
df = pd.read_csv('ufabc-score-predictor/dados/bd_prograd01_2023_sem_identificadores.csv')
df['nota'] = df['nota'].astype(str).str.replace(',', '.').astype(float)

# Definir os grupos
group_cols = ['modalidade inscrição', 'opção de turno', 'opção de campus']

# Agrupar e pegar a menor nota de cada grupo
menores_notas = (
    df.groupby(group_cols)['nota']
    .min()
    .reset_index()
    .sort_values(group_cols)
)

# Printar o resultado
for _, row in menores_notas.iterrows():
    print(
        f"Modalidade: {row['modalidade inscrição']} | "
        f"Turno: {row['opção de turno']} | "
        f"Campus: {row['opção de campus']} | "
        f"Menor nota: {row['nota']:.2f}"
    )