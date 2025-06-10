# UFABC Score Predictor

Este projeto utiliza modelos de Machine Learning para prever a nota e a probabilidade de aprovação de candidatos em processos seletivos da UFABC, considerando diferentes modalidades, turnos, campi e cursos.

## Estrutura do Projeto

```
src/
│
├── application/
│   └── app_streamlit.py         # Aplicação web para previsão (Streamlit)
│
├── data_prep/
│   └── data_drop.py             # Scripts de preparação e limpeza dos dados
│
├── train_models/
│   ├── train_xgboost.py                 # Treinamento do modelo de regressão (nota)
│   ├── train_xgboost_classifier.py      # Treinamento do classificador de aprovação
│   └── train_xgboost_classifier_grid.py # Treinamento do classificador com grid search
│
├── utils/
│   ├── menor_nota_convocado.py          # Utilitário para menor nota de convocados
│   └── menor_nota_por_grupo.py          # Utilitário para menor nota por grupo
│
└── README.md
requirements.txt
```

## Como funciona

- **Previsão de Nota:**  
  O modelo de regressão XGBoost é treinado para prever a nota de candidatos com base em modalidade de inscrição, turno, campus e curso. O treinamento é feito por curso, e os modelos são salvos em `modelos_xgboost/regressores_por_curso`.

- **Previsão de Aprovação:**  
  Um classificador XGBoost é treinado para prever a probabilidade de aprovação (status "Matriculado(a)") usando as mesmas variáveis e a nota prevista. O modelo e o transformer são salvos em `modelos_xgboost/classifier_aprovacao.pkl` e `classifier_transformer.pkl`.

- **Aplicação Web:**  
  O arquivo `src/application/app_streamlit.py` permite ao usuário selecionar modalidade, turno, campus e curso, e retorna a nota prevista, a probabilidade de aprovação e comparações com médias e menores notas dos convocados.

## Como rodar

1. **Instale as dependências:**
   ```
   pip install -r requirements.txt
   ```

2. **Prepare os dados:**
   - Execute scripts em `src/data_prep/` se necessário para limpar ou ajustar os dados.

3. **Treine os modelos:**
   - Para regressão (nota):
     ```
     python src/train_models/train_xgboost.py
     ```
   - Para classificação (probabilidade de aprovação):
     ```
     python src/train_models/train_xgboost_classifier.py
     ```

4. **Execute o aplicativo Streamlit:**
   ```
   streamlit run src/application/app_streamlit.py
   ```

## Observações

- Os modelos de regressão são treinados e salvos por curso, e o modelo de classificação é geral.
- O projeto utiliza pré-processamento com OneHotEncoder para variáveis categóricas.
- O código remove outliers das notas usando o método IQR.
- O cálculo da probabilidade de aprovação pode ser feito tanto pelo classificador quanto por uma função sigmoide suavizada baseada na diferença entre a nota prevista e a média dos convocados.

## Requisitos

- Python 3.8+
- xgboost
- scikit-learn
- pandas
- streamlit
- joblib
- numpy
- unidecode

Veja o arquivo `requirements.txt` para detalhes.

---

**Desenvolvido para fins acadêmicos e de apoio à decisão em processos seletivos.**