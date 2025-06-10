import pandas as pd

# 1. Carregar o arquivo CSV
df = pd.read_csv('dados/bd_prograd01_2023.csv', encoding='latin1', sep=';')

print(df.columns)  # Adicione esta linha para ver os nomes das colunas

# 2. Remover as colunas "inscrição no ENEM" e "nome candidato"
df = df.drop(['inscrição no ENEM', 'nome candidato'], axis=1)

# 3. (Opcional) Salvar o novo arquivo sem essas colunas
df.to_csv('dados/bd_prograd01_2023_sem_identificadores.csv', index=False)

# Agora df está pronto para os próximos passos, sem as colunas indesejadas.

print(f"Quantidade de linhas originais: {len(df)}")

def limpar_vazios(df):
    # Substitui strings vazias ou espaços por NaN
    df_temp = df.replace(r'^\s*$', pd.NA, regex=True)
    # Identifica linhas com qualquer valor vazio
    linhas_vazias = df_temp.isna().any(axis=1)
    qtd_removidas = linhas_vazias.sum()
    print(f"Quantidade de linhas a serem removidas por valores vazios: {qtd_removidas}")
    # Remove as linhas
    df_limpo = df_temp.dropna(how='any')
    return df_limpo

df = limpar_vazios(df)

# Supondo que a coluna se chama exatamente 'modalidade inscrição'
modalidades = df['modalidade inscrição'].unique()

# Dicionário para armazenar os DataFrames separados
dfs_por_modalidade = {}

for modalidade in modalidades:
    df_modalidade = df[df['modalidade inscrição'] == modalidade]
    dfs_por_modalidade[modalidade] = df_modalidade
    print(f"Modalidade: {modalidade} - {len(df_modalidade)} linhas")

# Agora você pode acessar cada cluster pelo nome da modalidade, por exemplo:
# dfs_por_modalidade['Ampla Concorrência']