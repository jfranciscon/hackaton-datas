import pandas as pd

def carregar_menores_notas_por_grupo(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    df['nota'] = df['nota'].astype(str).str.replace(',', '.').astype(float)
    df['status de matrícula'] = df['status de matrícula'].str.strip().str.lower()
    
    df_convocado = df[df['status de matrícula'] == 'matriculado(a)']
    
    group_cols = ['modalidade inscrição', 'opção de turno', 'opção de campus', 'opção de curso']
    menores_notas = (
        df_convocado
        .groupby(group_cols)['nota']
        .min()
        .reset_index()
        .sort_values(group_cols)
    )
    return menores_notas
