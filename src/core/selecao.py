"""Seleção do melhor modelo no melhor cenário, com dados retrospectivos.

A regra é aplicada por horizonte, somente com as previsões walk-forward até 31/08/2026:
vence a configuração de menor MAE, com o RMSE como desempate.

T+1 (preparo): modelos isolados nos cenários C1 a C4 e híbridos 50/50 nos cenários C1 a C3.
T+2 (compras): modelos isolados nos cenários C1 a C3, pois não há quantitativo institucional.
"""
import json
from datetime import date
from typing import Tuple

import numpy as np
import pandas as pd

from .config import RESULTADOS, SILVER

PASTA_MODELOS = RESULTADOS / "modelos"
ARQUIVO_SELECAO = PASTA_MODELOS / "selecao_modelos.json"
ARQUIVO_CANDIDATOS = PASTA_MODELOS / "selecao_candidatos_fev_ago.csv"
INICIO_RETROSPECTIVO = "2026-02-01"
FIM_RETROSPECTIVO = "2026-08-31"
FIM_TREINO_CONGELADO = "2026-09-01"
PESO_IA_HIBRIDO = 0.50
CENARIO_QUANTITATIVO = "Cenario_4_Com_Nutricionista"
PREFIXO_REFERENCIA = "Baseline_"
MESES_TREINO = 12
NOMES_MODELOS = {
    "Regressao_Linear": "Regressão Linear", "ElasticNet": "Elastic Net", "Ensemble_Bagging_RandomForest": "Random Forest",
    "Ensemble_Bagging_ExtraTrees": "ExtraTrees", "Ensemble_Bagging_Arvores": "Bagging de Árvores", "Arvore_Decisao": "Árvore de Decisão",
    "MLP": "Rede Neural MLP", "Ensemble_Boosting_Gradient": "Gradient Boosting", "Ensemble_Boosting_AdaBoost": "AdaBoost",
    "Ensemble_Boosting_LightGBM": "LightGBM", "Ensemble_Boosting_XGBoost": "XGBoost", "Ensemble_Boosting_CatBoost": "CatBoost",
    "Ensemble_Voting_Heterogeneo": "Voting", "Ensemble_Stacking_MetaModel": "Stacking", "Ensemble_Blending_Holdout": "Blending",
    "Ensemble_Cascading_Estagios": "Cascading"
}


def combinar_hibrido(pred_ia: np.ndarray, pred_nutri: np.ndarray, peso_ia: float = PESO_IA_HIBRIDO) -> np.ndarray:
    """Combina a previsão da IA e o plano da nutricionista, arredondando para refeição inteira."""
    return np.floor(peso_ia * np.asarray(pred_ia, dtype=float) + (1.0 - peso_ia) * np.asarray(pred_nutri, dtype=float) + 0.5)


def _metricas(y: np.ndarray, p: np.ndarray) -> dict:
    erro = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
    return {
        "n_dias": len(erro),
        "MAE": float(np.abs(erro).mean()),
        "RMSE": float(np.sqrt((erro ** 2).mean())),
        "WAPE_pct": float(np.abs(erro).sum() / np.sum(y) * 100.0),
        "R2": float(1.0 - (erro ** 2).sum() / ((y - np.mean(y)) ** 2).sum()),
        "Sobra": float(np.maximum(erro, 0).sum()),
        "Deficit": float(np.maximum(-erro, 0).sum())
    }


def previsoes_retrospectivas(preds: pd.DataFrame) -> pd.DataFrame:
    """Mantém apenas o período retrospectivo: nada de Set-Out chega à seleção."""
    datas = pd.to_datetime(preds["data"])
    return preds[(datas >= INICIO_RETROSPECTIVO) & (datas <= FIM_RETROSPECTIVO)].copy()


def avaliar_candidatos(preds: pd.DataFrame, planos: pd.Series) -> pd.DataFrame:
    """Calcula as métricas de cada candidato em Fev-Ago, para modelos isolados e híbridos."""
    retro = previsoes_retrospectivas(preds)
    retro = retro[~retro["algoritmo"].str.startswith(PREFIXO_REFERENCIA)]
    linhas = []
    for (horizonte, cenario, algoritmo), grupo in retro.groupby(["horizonte", "cenario", "algoritmo"]):
        y, p = grupo["consumo_real"].values.astype(float), grupo["previsao"].values.astype(float)
        base = {"horizonte": horizonte, "cenario": cenario, "algoritmo": algoritmo}
        linhas.append({**base, "tipo": "isolado", **_metricas(y, p)})
        if horizonte == "t1" and cenario != CENARIO_QUANTITATIVO:
            plano = pd.to_datetime(grupo["data"]).map(planos).values.astype(float)
            valido = ~np.isnan(plano)
            linhas.append({**base, "tipo": "hibrido", **_metricas(y[valido], combinar_hibrido(p[valido], plano[valido]))})
    return pd.DataFrame(linhas).sort_values(["horizonte", "MAE", "RMSE"]).reset_index(drop=True)


def escolher_vencedores(candidatos: pd.DataFrame) -> dict:
    """Aplica a regra: menor MAE, com RMSE como desempate, em cada horizonte."""
    vencedores = {}
    for horizonte in ("t1", "t2"):
        v = candidatos[candidatos["horizonte"] == horizonte].sort_values(["MAE", "RMSE"]).iloc[0]
        vencedores[horizonte] = {
            "algoritmo": v["algoritmo"], "cenario": v["cenario"], "tipo": v["tipo"],
            "MAE": round(float(v["MAE"]), 2), "RMSE": round(float(v["RMSE"]), 2), "n_dias": int(v["n_dias"])
        }
    return vencedores


def planos_da_silver() -> pd.Series:
    silver = pd.read_parquet(SILVER / "silver_fato_diaria.parquet")
    silver["data"] = pd.to_datetime(silver["data"])
    return silver.set_index("data")["plano_nutricionista"]


def executar_selecao() -> Tuple[pd.DataFrame, dict]:
    """Seleciona os modelos do preparo e das compras e grava o registro da decisão."""
    preds = pd.read_parquet(PASTA_MODELOS / "predicoes_detalhadas_torneio.parquet")
    candidatos = avaliar_candidatos(preds, planos_da_silver())
    candidatos.to_csv(ARQUIVO_CANDIDATOS, sep=";", index=False, encoding="utf-8-sig")
    registro = {
        "regra": "Menor MAE em Fev-Ago/2026 por horizonte, com RMSE como desempate",
        "periodo": [INICIO_RETROSPECTIVO, FIM_RETROSPECTIVO],
        "fim_do_treino_congelado": FIM_RETROSPECTIVO,
        "reprocessado_em": date.today().isoformat(),
        "vencedores": escolher_vencedores(candidatos)
    }
    ARQUIVO_SELECAO.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    return candidatos, registro


def carregar_selecao() -> dict:
    """Lê os modelos escolhidos no retrospectivo."""
    return json.loads(ARQUIVO_SELECAO.read_text(encoding="utf-8"))["vencedores"]


def nome_modelo(algoritmo: str) -> str:
    return NOMES_MODELOS.get(algoritmo, algoritmo)


def treinar_congelado(gold: pd.DataFrame, algoritmo: str, colunas: list):
    """Treina o modelo escolhido com a configuração do torneio e os 12 meses anteriores a 01/09/2026."""
    from .models import LIMITES_ALVO, obter_catalogo_modelos
    fim = pd.Timestamp(FIM_TREINO_CONGELADO)
    treino = gold[
        (gold["data"] < fim) & (gold["data"] >= fim - pd.DateOffset(months=MESES_TREINO)) &
        gold["usar_no_treino"] & gold["consumo"].notna() & gold["alvo_relativo"].notna()
    ]
    modelo = obter_catalogo_modelos()[algoritmo]["construtor"]()
    return modelo.fit(treino[colunas], treino["alvo_relativo"].clip(*LIMITES_ALVO))
