"""
Selecao de features e definicao do pipeline de pre-processamento para o
modelo supervisionado de alfabetizacao (Tech Challenge - Fase 3).

O ColumnTransformer aqui definido NAO e ajustado (fit) neste modulo -- ele
e combinado com o modelo dentro de um unico sklearn.Pipeline nas etapas
seguintes (treinamento, otimizacao de hiperparametros). Isso garante que,
durante cross-validation, o imputer/scaler/encoder sao reajustados a cada
fold usando so os dados de treino daquele fold, sem vazar informacao do
fold de validacao -- o mesmo cuidado de leakage da Parte 1, agora aplicado
ao pre-processamento.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUNAS_IDENTIFICADORAS = ["id_municipio", "id_escola", "id_aluno"]
COLUNA_TARGET_TEXTO = "alfabetizado"  # versao string; o target usado e alfabetizado_flag
COLUNA_TARGET = "alfabetizado_flag"
COLUNA_PESO = "peso_aluno"  # peso amostral -- nao e feature, fica reservado a parte

COLUNAS_CATEGORICAS = ["rede", "sigla_uf", "presenca", "preenchimento_caderno", "caderno", "ano"]

COLUNAS_NUMERICAS = [
    "taxa_alfabetizacao_municipio_ano_anterior",
    "media_portugues_municipio_ano_anterior",
    "taxa_alfabetizacao_uf_ano_anterior",
    "media_portugues_uf_ano_anterior",
] + [f"meta_alfabetizacao_{ano}" for ano in range(2024, 2031)] \
  + [f"meta_alfabetizacao_{ano}_uf" for ano in range(2024, 2031)]


def preparar_x_y(df: pd.DataFrame):
    """Separa a base Gold em X (features), y (target) e w (peso amostral)."""
    colunas_a_remover = COLUNAS_IDENTIFICADORAS + [COLUNA_TARGET_TEXTO, COLUNA_TARGET, COLUNA_PESO]
    colunas_features = COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS

    faltando = set(colunas_features) - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas esperadas ausentes na base: {faltando}")

    x = df[colunas_features].copy()
    x["ano"] = x["ano"].astype(str)  # ano tratado como categoria, nao como numero continuo
    y = df[COLUNA_TARGET].copy()
    w = df[COLUNA_PESO].copy()
    return x, y, w


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer com imputacao + transformacao, para numericas e categoricas."""
    pipeline_numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pipeline_categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("numerico", pipeline_numerico, COLUNAS_NUMERICAS),
        ("categorico", pipeline_categorico, COLUNAS_CATEGORICAS),
    ])
