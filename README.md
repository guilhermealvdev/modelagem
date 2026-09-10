Modelagem — Predição e Inteligência Analítica para Alfabetização no Brasil

Tech Challenge — Fase 3 (Pós-Tech Data Analytics). Modelo supervisionado de classificação para prever se um aluno será considerado alfabetizado, usando dados educacionais, territoriais e socioeconômicos derivados do Indicador Criança Alfabetizada.

Repositório da Fase 2 (origem dos dados): https://github.com/guilhermealvdev/pipeline-hibrido

1. Contexto do problema

A alfabetização na infância é um dos pilares do desenvolvimento educacional, social e econômico de um país. O Compromisso Nacional Criança Alfabetizada mobiliza União, estados, Distrito Federal e municípios para garantir que todas as crianças brasileiras estejam alfabetizadas até o final do 2º ano do ensino fundamental, com meta nacional de 100% até 2030.

Entender apenas os dados atuais não é suficiente para apoiar decisões estratégicas — gestores públicos precisam antecipar risco, identificar regiões vulneráveis e entender quais fatores mais pesam nos indicadores educacionais. Este projeto usa a camada Gold construída na Fase 2 para treinar um modelo capaz de prever, por aluno, se ele será considerado alfabetizado, e usa esse modelo para gerar inteligência aplicada a essas perguntas.

2. Objetivo analítico

Desenvolver um modelo supervisionado de classificação binária que prevê se um aluno será alfabetizado (Sim/Não), usando variáveis educacionais e territoriais — e usar o modelo (interpretabilidade + análises descritivas) para responder:

Quais fatores mais impactam a alfabetização?
Quais municípios apresentam maior risco educacional?
Quais regiões possuem padrões semelhantes?
Quais municípios estão mais distantes de cumprir a meta nacional de 2030?

3. Descrição da base utilizada

A base parte da camada Silver da Fase 2 (alfabetizacao_integrado.parquet, 3.867.999 alunos, 2023-2024) — é a única camada com granularidade por aluno; a Gold original da Fase 2 é agregada por município e não serve para este problema. Construímos uma nova camada Gold, no nível de aluno (base_alunos_modelagem.parquet), juntando:

Dados do aluno: rede de ensino, presença, preenchimento da prova, ano
Contexto do município e da UF do ano anterior (taxa de alfabetização histórica, média de português) — defasado em 1 ano deliberadamente, para não vazar o resultado do ano corrente
Metas oficiais de alfabetização (município e UF, 2024-2030)

Target: alfabetizado (Sim/Não), praticamente balanceado (~51%/49%, estável entre 2023 e 2024).

Achado crítico de data leakage: a variável proficiencia determina alfabetizado por um corte exato em ~743 pontos (a própria regra de negócio do indicador) — foi excluída do conjunto de features. A variável serie também foi excluída por ser constante em toda a base.

4. Etapas de modelagem

Split treino/teste (80/20, estratificado, seed fixa) — o teste só foi tocado uma única vez, ao final da otimização de hiperparâmetros.
Pré-processamento integrado num único sklearn.Pipeline: imputação (mediana para numéricas, moda para categóricas) + scaling (StandardScaler) + encoding (OneHotEncoder) — sempre ajustado só no treino de cada fold, nunca na base inteira, para evitar vazamento entre treino e validação/teste.
Comparação de modelos via cross-validation (3 folds) no treino: Regressão Logística vs Random Forest.
Otimização de hiperparâmetros da Random Forest via Grid Search + cross-validation (n_estimators, max_depth), com checagem explícita de overfitting comparando score de treino vs validação em cada combinação.
Avaliação final no teste, feita uma única vez.
Interpretabilidade: Feature Importance nativa + SHAP (amostra de 2.000 alunos do teste).

5. Escolha do algoritmo

Modelo	Accuracy	Precision	Recall	F1	ROC-AUC
Regressão Logística	0.685	0.653	0.827	0.729	0.757
Random Forest	0.687	0.669	0.772	0.716	0.760

A Random Forest venceu por ROC-AUC (métrica mais robusta a balanceamento de classe), com desvio-padrão entre folds pequeno o suficiente (~0.0004) para confiar que a diferença é real, não ruído. A Regressão Logística teve recall mais alto — vale mencionar como trade-off: para um problema de identificação de risco educacional, recall alto (menos falsos negativos) tem valor prático, mesmo o critério técnico tendo escolhido a Random Forest.

Após o Grid Search (n_estimators=200, max_depth=15), o ROC-AUC de validação subiu para 0.768.

6. Métricas de avaliação

Avaliação final no teste (nunca visto antes pelo modelo):

Métrica	Valor
Accuracy	0.692
Precision	0.659
Recall	0.830
F1	0.734
ROC-AUC	0.768

Matriz de confusão:

	Previsto: Não	Previsto: Sim
Real: Não	205.805	170.886
Real: Sim	67.453	329.456

O modelo erra mais para o lado de falso positivo — tende a ser mais otimista, classificando mais alunos como alfabetizados do que deveria.

Checagem de generalização: ROC-AUC de validação (0.768) e de teste (0.768) praticamente idênticos — o desempenho não foi um acaso do cross-validation, se sustenta em dado nunca visto. Sem sinais de overfitting.

7. Interpretação dos resultados

Feature Importance e SHAP concordam: preenchimento_caderno e presença dominam a importância (~0.15-0.18 cada, muito acima de qualquer outra variável) — refletem a regra determinística de que um aluno ausente ou com prova não preenchida é automaticamente registrado como não-alfabetizado. Depois dessas duas, as metas de alfabetização (2024-2029) e o contexto municipal/UF do ano anterior aparecem com peso moderado — indicando que, entre os alunos que de fato realizaram a avaliação, o histórico da região pesa mais do que qualquer característica isolada do aluno.

8. Insights encontrados

Fatores mais relevantes: presença/preenchimento da prova (efeito mecânico), seguidos pelas metas oficiais e pelo contexto histórico do município/UF.
Municípios de maior risco: concentrados majoritariamente na Bahia e no Rio Grande do Norte, com taxas de alfabetização observadas entre 7,9% e 15,4% (após filtrar municípios com baixa cobertura de dados — ver Limitações).
Padrão regional: Norte tem a menor taxa de alfabetização (41,8%), Centro-Oeste a maior (54,7%) — Nordeste, Sul e Sudeste ficam numa faixa intermediária (50,6% a 53,6%).
Distância da meta 2030: os municípios mais distantes da meta nacional de 80% precisariam de saltos de 68 a 76 pontos percentuais na taxa de alfabetização — praticamente inviável no ritmo atual sem intervenção significativa.
Achado atípico: o Ceará aparece com taxa de alfabetização de 82%, muito acima do segundo colocado (Espírito Santo, 63%) — vale investigação adicional para confirmar se é um resultado genuíno ou uma particularidade de composição de rede/amostra.

9. Limitações do projeto

Qualidade de dado por município: ao menos um município (GO, 410 alunos) teve 100% dos registros marcados como "ausente" em 2023 — não é risco educacional real, é ausência de aplicação/registro da prova. O ranking de risco final filtra por taxa de presença mínima (≥50%) para mitigar isso, mas pode haver casos menos extremos não filtrados.
Contexto histórico ausente em 2023: por defasarmos o contexto municipal/UF em 1 ano (decisão deliberada contra leakage), os alunos de 2023 não têm esse contexto disponível (imputado pela mediana).
Poder preditivo real é moderado: descontando o efeito mecânico de presença/preenchimento, o modelo captura sinal real mas não muito forte (ROC-AUC ~0.77) — alfabetização depende de fatores não capturados nesta base (qualidade pedagógica da escola, perfil socioeconômico familiar detalhado, etc.).
Metas como feature: as metas oficiais de alfabetização correlacionam com o resultado, mas isso pode refletir que metas foram fixadas com base no desempenho histórico da região — é uma correlação válida para o modelo prever, mas não deve ser lida como "a meta causa o resultado".

10. Aplicação prática para políticas públicas

O ranking de municípios de maior risco (Seção 8) pode orientar priorização de investimento e acompanhamento pedagógico.
A comparação regional aponta o Norte como a região que precisa de atenção mais urgente e ampla, não só municípios isolados.
A lista de municípios mais distantes da meta 2030 permite antecipar onde o Compromisso Nacional Criança Alfabetizada está mais em risco de não ser cumprido, dando tempo para intervenção.
O achado sobre qualidade de dado (municípios com prova não aplicada) é, por si, um insight operacional: antes de qualquer política de alfabetização, é preciso garantir que a avaliação está sendo aplicada de fato.

11. Possíveis evoluções futuras

Enriquecer com fontes externas sugeridas no enunciado (IBGE, Censo Escolar, FUNDEB, PNAD, Atlas do Desenvolvimento Humano) para capturar fatores socioeconômicos mais diretos.
Investigar separadamente o subconjunto de alunos que de fato realizou a prova (excluindo o efeito mecânico de presença/preenchimento), para isolar o sinal pedagógico real.
Modelar clusterização de municípios (K-Means/hierárquico) para a pergunta de "regiões com padrões semelhantes" de forma mais rigorosa do que a agregação por UF/região feita aqui.
Investigar a fundo o outlier do Ceará antes de comunicar o achado.
Ampliar o grid de hiperparâmetros (mais valores, mais parâmetros) com mais tempo/capacidade computacional disponível.


Estrutura do repositório

modelagem/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── model_input/          # split treino/teste + modelo final (fora do git)
├── notebooks/
│   ├── 01_preparacao_base_aluno.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessamento.ipynb
│   ├── 04_treinamento_modelos.ipynb
│   ├── 05_otimizacao.ipynb
│   └── 06_interpretabilidade.ipynb
├── src/
│   ├── preprocessing/
│   │   └── pipeline.py
│   └── processing/
│       └── gold/
│           └── construir_gold_alunos.py
└── reports/
    ├── comparacao_modelos.csv
    └── roteiro_video_executivo.md


Como rodar
bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install matplotlib scikit-learn shap   # nao incluidos no requirements.txt original

# Rodar os notebooks em ordem, 01 a 06 (Run All em cada um)
