"""Garante que a escolha dos modelos usa só o retrospectivo e que o congelamento para em 31/08/2026."""
import numpy as np
import pandas as pd

from src.core.config import GOLD, RESULTADOS
from src.core import selecao
from src.core.models import definir_cenarios_features

PREDS = RESULTADOS / "modelos" / "predicoes_detalhadas_torneio.parquet"


def test_escolha_nao_muda_com_set_out_corrompido():
    preds = pd.read_parquet(PREDS); planos = selecao.planos_da_silver()
    original = selecao.escolher_vencedores(selecao.avaliar_candidatos(preds, planos))
    corrompido = preds.copy()
    futuro = pd.to_datetime(corrompido["data"]) >= "2026-09-01"
    corrompido.loc[futuro, "previsao"] = corrompido.loc[futuro, "consumo_real"]
    assert selecao.escolher_vencedores(selecao.avaliar_candidatos(corrompido, planos)) == original


def test_compras_nao_concorrem_com_quantitativo():
    candidatos = selecao.avaliar_candidatos(pd.read_parquet(PREDS), selecao.planos_da_silver())
    t2 = candidatos[candidatos["horizonte"] == "t2"]
    assert selecao.CENARIO_QUANTITATIVO not in set(t2["cenario"])
    assert set(t2["tipo"]) == {"isolado"}
    assert selecao.carregar_selecao()["t2"]["cenario"] != selecao.CENARIO_QUANTITATIVO


def test_registro_corresponde_a_regra():
    candidatos = selecao.avaliar_candidatos(pd.read_parquet(PREDS), selecao.planos_da_silver())
    registro = selecao.carregar_selecao()
    for horizonte, vencedor in selecao.escolher_vencedores(candidatos).items():
        assert registro[horizonte] == vencedor


def test_congelamento_nao_usa_dados_de_setembro(monkeypatch):
    gold = pd.read_parquet(GOLD / "gold_t2.parquet")
    ref = selecao.carregar_selecao()["t2"]
    colunas = [c for c in definir_cenarios_features()[ref["cenario"]] if c in gold.columns]
    linha = gold[(gold["data"] >= "2026-09-01") & gold["consumo"].notna()].head(5)
    base = selecao.treinar_congelado(gold, ref["algoritmo"], colunas).predict(linha[colunas])
    alterado = gold.copy()
    futuro = alterado["data"] >= "2026-09-01"
    alterado.loc[futuro, "alvo_relativo"] = 2.0
    alterado.loc[futuro, "consumo"] = 500.0
    assert np.array_equal(base, selecao.treinar_congelado(alterado, ref["algoritmo"], colunas).predict(linha[colunas]))
