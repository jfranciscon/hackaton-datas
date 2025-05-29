import pandas as pd
import matplotlib.pyplot as plt

# Leitura da chamada regular do SISU 2023, com leitura segura forçada
cr_sisu = pd.read_csv("../banco_dados_hackaton/chamada_regular_sisu_2023_1.csv", sep='|', encoding='latin1', low_memory=False)
# Leitura da lista de espera do SISU 2023, com leitura segura forçada
le_sisu = pd.read_csv("../banco_dados_hackaton/lista_de_espera_sisu_2023_1.csv", sep='|', encoding='latin1', low_memory=False)
# Leitura do BD da Prograd, 1a chamada de 2023, com leitura segura forçada
bd_prograd = pd.read_csv("../banco_dados_hackaton/bd_prograd01_2023.csv", sep='|', encoding='latin1', low_memory=False)

# Visualizar as primeiras linhas
print(cr_sisu.head())

# Visualizar o nome das colunas
print(cr_sisu.columns.tolist())

# Converter a nota do candidato e nota de corte para float
cr_sisu["NOTA_CANDIDATO"] = cr_sisu["NOTA_CANDIDATO"].str.replace(",", ".", regex=False).astype(float)
cr_sisu["NOTA_CORTE"] = cr_sisu["NOTA_CORTE"].str.replace(",", ".", regex=False).astype(float)

# Filtrar para apenas UFABC
cr_sisu_ufabc = cr_sisu[cr_sisu["NOME_IES"].str.contains("FUNDAÇÃO UNIVERSIDADE FEDERAL DO ABC", case=False, na=False)]

# Ver cursos oferecidos (ver como o BC&T está escrito)
print(cr_sisu_ufabc["NOME_CURSO"].unique())

# Média das Notas Por Curso
media_por_curso = cr_sisu_ufabc.groupby("NOME_CURSO")["NOTA_CANDIDATO"].mean().sort_values(ascending=False)
print(media_por_curso)

# Filtrar o curso de Ciência e Tecnologia
cr_sisu_ufabc_bct = cr_sisu_ufabc[cr_sisu_ufabc["NOME_CURSO"].str.contains("CIÊNCIA E TECNOLOGIA", case=False, na=False)]

# Ver distribuição por turno
print(cr_sisu_ufabc_bct["TURNO"].value_counts())

# Ver média da nota por turno
print(cr_sisu_ufabc_bct.groupby("TURNO")["NOTA_CANDIDATO"].mean())

# Ver média da nota de corte por modalidade
print(cr_sisu_ufabc_bct.groupby("MOD_CONCORRENCIA")["NOTA_CORTE"].mean())


# Gráfico de barras de Comparação Por Turno
import matplotlib.pyplot as plt

cr_sisu_ufabc_bct.groupby("TURNO")["NOTA_CANDIDATO"].mean().plot(kind="bar")
plt.title("Média da Nota por Turno – BCT UFABC")
plt.ylabel("Nota do Candidato")
plt.xlabel("Turno")
plt.tight_layout()
plt.show()

# Gráfico de barras de média das notas por curso (top 10)
import matplotlib.pyplot as plt

media_por_curso.head(10).plot(kind="barh")
plt.title("Top 10 Cursos da UFABC com Maior Média de Nota")
plt.xlabel("Nota Média dos Candidatos")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


