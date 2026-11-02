"""Modelos auditados servidos pelo aplicativo.

T+1 usa o Híbrido 50/50 (ExtraTrees Cenário 2 + quantitativo institucional) e T+2 usa o ExtraTrees
Cenário 2, sem quantitativo. Os modelos ficam congelados com dados até 31/08/2026, com a especificação
da pesquisa. No simulador, datas anteriores a 01/09/2026 são refeitas em walk-forward até a sexta de corte.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import streamlit as st
    cache_resource = st.cache_resource
    cache_data = st.cache_data(ttl=1800)
except Exception:
    def cache_resource(fn):
        return fn
    def cache_data(fn):
        return fn

from src.core.bronze_silver import padronizar_texto
from src.core.config import GOLD, HORIZONTES, RESULTADOS, SEMENTE_ALEATORIA
from src.core.gold_features import construir_emissao, origem_da_janela
from src.core.safety_margins import FIM_TREINO_CONGELADO, combinar_hibrido

CENARIOS_CORE = {
    1: "Cenario_1_Base_Cardapio",
    2: "Cenario_2_Com_Cardapio",
    3: "Cenario_3_Com_Clima",
    4: "Cenario_4_Com_Nutricionista"
}
QUANTIS = (("P75", 0.75), ("P85", 0.85), ("P90", 0.90))
LIMITES_ALVO = (0.2, 2.5)

MAPA_PROTEINA = {"bovina": "bovina", "frango": "frango", "suina": "suina", "peixe": "peixe", "feijoada": "feijoada", "mista": "mista"}
MAPA_PREPARO = {
    "ao molho": "ao_molho_ou_refogado",
    "assado": "assado",
    "churrasco": "churrasco_ou_espeto",
    "cozido": "cozido_ou_refogado",
    "cremoso": "cremoso_ou_ensopado",
    "ensopado": "cremoso_ou_ensopado",
    "empanado": "empanado_ou_a_milanesa",
    "frito": "frito",
    "grelhado": "grelhado",
    "massa": "massa_ou_montado"
}


def _normalizar(mapa: dict, valor) -> object:
    """Converte o texto cadastrado no app para a categoria usada na camada Gold."""
    if pd.isna(valor):
        return np.nan
    texto = str(padronizar_texto(valor)).lower()
    return next((alvo for chave, alvo in mapa.items() if chave in texto), np.nan)


def preparar_fato(df_fato: pd.DataFrame) -> pd.DataFrame:
    """Adapta a tabela fato do app, com edições operacionais, ao formato esperado pela camada Gold."""
    fato = df_fato.copy()
    fato["data"] = pd.to_datetime(fato["data_iso"])
    fato = fato.drop_duplicates("data", keep="last").sort_values("data").reset_index(drop=True)
    fato["consumo"] = pd.to_numeric(fato["consumo"], errors="coerce").where(lambda s: s > 0)
    fato["tem_consumo"] = fato["consumo"].notna()
    fato["dia_semana"] = fato["data"].dt.dayofweek
    fato["no_escopo"] = fato["dia_semana"] < 5

    # Pratos cadastrados no app substituem o cardápio da base
    cadastrado = fato["prato_principal"].notna() & fato["prato_principal"].ne(fato["prato"])
    if cadastrado.any():
        fato.loc[cadastrado, "prato"] = fato.loc[cadastrado, "prato_principal"]
        fato.loc[cadastrado, "prato_padronizado"] = fato.loc[cadastrado, "prato_principal"].map(padronizar_texto)
        fato.loc[cadastrado, "proteina"] = fato.loc[cadastrado, "proteina_tipo"].map(lambda v: _normalizar(MAPA_PROTEINA, v))
        fato.loc[cadastrado, "preparo"] = fato.loc[cadastrado, "preparo_tipo"].map(lambda v: _normalizar(MAPA_PREPARO, v))
    return fato


@cache_data
def carregar_gold(horizonte: str) -> pd.DataFrame:
    """Lê a camada Gold auditada do horizonte e recria o atributo de quantitativo institucional."""
    gold = pd.read_parquet(GOLD / f"gold_{horizonte}.parquet")
    if horizonte == "t1":
        plano = gold["plano_nutricionista"].fillna(gold["reserva_discente"])
        gold["rel_plano_humano"] = (plano / gold["nivel_base"]).fillna(1.0)
    return gold


def colunas_do_cenario(horizonte: str, cenario: int) -> list:
    from src.core.models import definir_cenarios_features
    gold = carregar_gold(horizonte)
    return [c for c in definir_cenarios_features()[CENARIOS_CORE[cenario]] if c in gold.columns]


@cache_resource
def modelo_congelado(horizonte: str, cenario: int):
    """Treina o ExtraTrees congelado da pesquisa com dados até 31/08/2026."""
    from sklearn.ensemble import ExtraTreesRegressor
    from src.core.models import criar_pipeline_arvore
    gold = carregar_gold(horizonte)
    cols = colunas_do_cenario(horizonte, cenario)
    treino = gold[(gold["data"] < FIM_TREINO_CONGELADO) & (gold["ano"] >= 2025) & gold["usar_no_treino"] & gold["consumo"].notna()]
    modelo = criar_pipeline_arvore(ExtraTreesRegressor(n_estimators=80, min_samples_leaf=5, random_state=SEMENTE_ALEATORIA, n_jobs=1))
    return modelo.fit(treino[cols], treino["alvo_relativo"].clip(*LIMITES_ALVO))


@cache_resource
def modelo_walk_forward(horizonte: str, cenario: int, origem: pd.Timestamp):
    """Treina o ExtraTrees do torneio apenas com dias anteriores à sexta de corte."""
    from src.core.models import obter_catalogo_modelos
    gold = carregar_gold(horizonte)
    cols = colunas_do_cenario(horizonte, cenario)
    treino = gold[
        (gold["data"] < origem) & (gold["data"] >= origem - pd.DateOffset(months=12)) &
        gold["usar_no_treino"] & gold["consumo"].notna() & gold["alvo_relativo"].notna()
    ]
    if len(treino) < 20:
        return None
    modelo = obter_catalogo_modelos()["Ensemble_Bagging_ExtraTrees"]["construtor"]()
    return modelo.fit(treino[cols], treino["alvo_relativo"].clip(*LIMITES_ALVO))


def prever(modelo, linha: pd.DataFrame, horizonte: str, cenario: int) -> float:
    """Converte a previsão relativa do modelo em refeições inteiras."""
    cols = colunas_do_cenario(horizonte, cenario)
    relativa = np.clip(modelo.predict(linha.reindex(columns=cols)), *LIMITES_ALVO)
    return float(np.floor(relativa[0] * linha["nivel_base"].iloc[0] + 0.5))


def origem_da_data(data: pd.Timestamp, horizonte: str) -> pd.Timestamp:
    return origem_da_janela(pd.Series([pd.Timestamp(data)]), HORIZONTES[horizonte]["deslocamento_dias"]).iloc[0]


@cache_data
def atributos_da_emissao(df_fato: pd.DataFrame, origem: pd.Timestamp, horizonte: str) -> pd.DataFrame:
    """Calcula os atributos de uma emissão futura com as mesmas funções da camada Gold."""
    return construir_emissao(preparar_fato(df_fato), origem, horizonte)


def linha_arquivada(data: pd.Timestamp, horizonte: str) -> pd.DataFrame:
    """Devolve a linha da Gold, com a previsão de clima arquivada na emissão, quando a data existir."""
    gold = carregar_gold(horizonte)
    return gold[gold["data"] == pd.Timestamp(data)]


@cache_data
def carregar_margens() -> dict:
    """Lê as margens por quantil calibradas em Fev-Ago e deriva as do T+2 pelo mesmo método."""
    tabela = pd.read_csv(RESULTADOS / "tabelas_dissertacao" / "tabela_09_margens_empiricas_quantis.csv", sep=";", encoding="utf-8-sig")
    preds = pd.read_parquet(RESULTADOS / "modelos" / "predicoes_detalhadas_torneio.parquet")
    t2 = preds[
        (preds["horizonte"] == "t2") & (preds["cenario"] == CENARIOS_CORE[2]) & (preds["algoritmo"] == "Ensemble_Bagging_ExtraTrees") &
        (preds["data"] >= "2026-02-01") & (preds["data"] <= "2026-08-31")
    ]
    residuos = t2["consumo_real"] - t2["previsao"]
    margens = {}
    for nome, q in QUANTIS:
        ia = tabela[(tabela["Politica"] == "IA Pura (ExtraTrees)") & (tabela["Quantil"] == nome)].iloc[0]
        hib = tabela[(tabela["Politica"] == "Modelo Híbrido 50/50") & (tabela["Quantil"] == nome)].iloc[0]
        margens[nome] = {
            "ia": float(ia["Margem_ref"]),
            "hibrido": float(hib["Margem_ref"]),
            "t2": float(max(0.0, np.ceil(np.quantile(residuos, q)))),
            "cobertura_hibrido": float(hib["Cobertura_FevAgo_pct"]),
            "cobertura_ia": float(ia["Cobertura_FevAgo_pct"])
        }
    return margens


__all__ = [
    "atributos_da_emissao", "carregar_gold", "carregar_margens", "colunas_do_cenario", "combinar_hibrido", "FIM_TREINO_CONGELADO",
    "linha_arquivada", "modelo_congelado", "modelo_walk_forward", "origem_da_data", "prever"
]
