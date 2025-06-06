import pandas as pd

# Caminho ajustado para rodar a partir da raiz do projeto
df = pd.read_csv('ufabc-score-predictor/dados/bd_prograd01_2023_sem_identificadores.csv')
df['nota'] = df['nota'].astype(str).str.replace(',', '.').astype(float)

# Diagnóstico: veja os valores únicos
print(df['status de matrícula'].unique())

# Normalizar e filtrar
df['status de matrícula'] = df['status de matrícula'].str.strip().str.lower()
df_nao_convocado = df[df['status de matrícula'] == 'não convocado(a)']

# Agrupar e pegar a maior nota de cada grupo
group_cols = ['modalidade inscrição', 'opção de turno', 'opção de campus']
maiores_notas = (
    df_nao_convocado
    .groupby(group_cols)['nota']
    .max()
    .reset_index()
    .sort_values(group_cols)
)

# Printar o resultado
for _, row in maiores_notas.iterrows():
    print(
        f"Modalidade: {row['modalidade inscrição']} | "
        f"Turno: {row['opção de turno']} | "
        f"Campus: {row['opção de campus']} | "
        f"Maior nota (Não convocado): {row['nota']:.2f}"
    )

input("Pressione Enter para sair...")