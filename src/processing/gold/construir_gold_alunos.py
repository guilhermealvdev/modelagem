# -*- coding: utf-8 -*-
"""
Camada GOLD (nivel aluno) para o Tech Challenge - Fase 3.

Constroi uma tabela Gold com granularidade por aluno, pronta para
treinar um modelo supervisionado de classificacao (alfabetizado: Sim/Nao).

"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import BRONZE_DIR, GOLD_DIR, SILVER_DIR

#Prefixo do codigo IBGE do municipio (sigla das UFs)
PREFIXO_UF = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP",
    "17": "TO", "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB",
    "26": "PE", "27": "AL", "28": "SE", "29": "BA", "31": "MG", "32": "ES",
    "33": "RJ", "35": "SP", "41": "PR", "42": "SC", "43": "RS", "50": "MS",
    "51": "MT", "52": "GO", "53": "DF",
}


def carregar_bronze_latest(nome_entidade: str) -> pd.DataFrame:
    entity_dir = BRONZE_DIR / nome_entidade
    latest = sorted(entity_dir.glob("ingestion_date=*"))[-1]
    arquivo = next(latest.glob("*.parquet"))
    df = pd.read_parquet(arquivo)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def carregar_silver() -> pd.DataFrame:
    return pd.read_parquet(SILVER_DIR / "alfabetizacao_integrado.parquet")


def preparar_base_aluno(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["id_municipio"] = df["id_municipio"].astype(str).str.strip().str.zfill(7)
    df["sigla_uf"] = df["id_municipio"].str[:2].map(PREFIXO_UF)
    df["alfabetizado_flag"] = (df["alfabetizado"] == "Sim").astype(int)

    colunas_removidas = ["proficiencia", "serie", "id_municipio_nome", "_ingestion_date", "_source"]
    df = df.drop(columns=[c for c in colunas_removidas if c in df.columns])
    return df


def contexto_defasado(df_indicador: pd.DataFrame, chave: list, sufixo: str) -> pd.DataFrame:
    chave_completa = chave + ["rede"]
    colunas_indicador = [
        c for c in df_indicador.columns
        if c not in chave_completa + ["_ingestion_date", "_source", "serie"]
        and not c.endswith("_nome")
        and not c.startswith("proporcao_aluno_nivel_")
    ]
    df = df_indicador[df_indicador["rede"].isin(["Municipal", "Estadual"])].copy()
    df = df[chave_completa + colunas_indicador]
    df["ano"] = df["ano"] + 1  # essse indicador passa a representar o ano anterior
    df = df.rename(columns={c: f"{c}{sufixo}" for c in colunas_indicador})
    return df


def construir_gold_alunos() -> pd.DataFrame:
    silver = carregar_silver()
    base = preparar_base_aluno(silver)

    municipio = carregar_bronze_latest("municipio")
    uf = carregar_bronze_latest("uf")
    meta_municipio = carregar_bronze_latest("meta_alfabetizacao_municipio")
    meta_uf = carregar_bronze_latest("meta_alfabetizacao_uf")

    municipio["id_municipio"] = municipio["id_municipio"].astype(str).str.strip().str.zfill(7)
    meta_municipio["id_municipio"] = meta_municipio["id_municipio"].astype(str).str.strip().str.zfill(7)

    ctx_municipio = contexto_defasado(municipio, ["id_municipio", "ano"], "_municipio_ano_anterior")
    ctx_uf = contexto_defasado(uf, ["sigla_uf", "ano"], "_uf_ano_anterior")

    assert ctx_municipio.duplicated(subset=["id_municipio", "ano", "rede"]).sum() == 0
    assert ctx_uf.duplicated(subset=["sigla_uf", "ano", "rede"]).sum() == 0

    colunas_meta = [c for c in meta_municipio.columns if c.startswith("meta_alfabetizacao_")]
    meta_municipio_reduzida = meta_municipio[["id_municipio", "ano"] + colunas_meta].drop_duplicates()
    meta_uf_reduzida = meta_uf[["sigla_uf", "ano"] + colunas_meta].drop_duplicates(
    ).rename(columns={c: f"{c}_uf" for c in colunas_meta})

    assert meta_municipio_reduzida.duplicated(subset=["id_municipio", "ano"]).sum() == 0
    assert meta_uf_reduzida.duplicated(subset=["sigla_uf", "ano"]).sum() == 0

    gold = (
        base
        .merge(ctx_municipio, on=["id_municipio", "ano", "rede"], how="left")
        .merge(ctx_uf, on=["sigla_uf", "ano", "rede"], how="left")
        .merge(meta_municipio_reduzida, on=["id_municipio", "ano"], how="left")
        .merge(meta_uf_reduzida, on=["sigla_uf", "ano"], how="left")
    )
    return gold


def run():
    gold = construir_gold_alunos()
    caminho = GOLD_DIR / "base_alunos_modelagem.parquet"
    gold.to_parquet(caminho, index=False)
    print(f"[base_alunos_modelagem] salvo em {caminho} ({len(gold)} linhas, {gold.shape[1]} colunas)")
    print("Colunas:", gold.columns.tolist())


if __name__ == "__main__":
    run()