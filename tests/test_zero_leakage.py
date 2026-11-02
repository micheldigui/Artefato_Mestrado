import numpy as np
import pandas as pd
import pytest

from src.core.config import GOLD
from src.core.models import LIMITES_ALVO, arredondar_refeicoes, definir_cenarios_features
from src.core import test_zero_leakage_models as auditoria


@pytest.fixture(scope="module")
def golds():
    return pd.read_parquet(GOLD / "gold_t1.parquet"), pd.read_parquet(GOLD / "gold_t2.parquet")


def avaliador_com_defeito(filtro_alvo, filtro_treino):
    """Cria um avaliador igual ao real, com os filtros de teste e treino injetados."""
    def avaliador(nome_alg, info_alg, horizonte, nome_cenario, colunas, gold, origens):
        previsoes = []
        for origem in origens:
            alvo = gold[filtro_alvo(gold, origem)]
            treino = gold[filtro_treino(gold, origem)]
            if alvo.empty or len(treino) < 20:
                continue
            modelo = info_alg["construtor"]()
            modelo.fit(treino[colunas], treino["alvo_relativo"].clip(*LIMITES_ALVO))
            relativa = np.clip(modelo.predict(alvo[colunas]), *LIMITES_ALVO)
            previsoes += [{"previsao": p} for p in arredondar_refeicoes(relativa * alvo["nivel_base"].values)]
        return {}, previsoes
    return avaliador


def alvo_real(gold, origem):
    return (gold["origem"] == origem) & gold["consumo"].notna()


def treino_real(gold, origem):
    return (gold["data"] < origem) & gold["usar_no_treino"] & gold["consumo"].notna() & gold["alvo_relativo"].notna()


def test_auditoria_completa():
    rel = auditoria.executar_auditoria_completa_vazamento()
    assert rel["status_geral"] == "APROVADO NOS TESTES EXECUTADOS"
    for t in rel["testes"]:
        assert t["sucesso"] is True, f"Falha no teste: {t['teste']} - Erros: {t.get('erros')}"


def test_filtro_por_resultado_no_teste_e_detectado(golds):
    com_filtro = avaliador_com_defeito(lambda g, o: alvo_real(g, o) & g["usar_no_treino"], treino_real)
    assert auditoria.testar_selecao_do_teste(golds[0], com_filtro)["sucesso"] is False


def test_treino_ate_a_sexta_de_corte_e_detectado(golds):
    def treino_errado(gold, origem):
        return (gold["data"] <= origem) & gold["usar_no_treino"] & gold["consumo"].notna() & gold["alvo_relativo"].notna()
    corte_errado = avaliador_com_defeito(alvo_real, treino_errado)
    assert auditoria.testar_fronteira_temporal_todas_semanas(*golds, corte_errado)["sucesso"] is False


@pytest.mark.parametrize("coluna", ["consumo", "oraculo_chuva_mm", "usar_no_treino"])
def test_coluna_proibida_em_x_e_detectada(golds, coluna):
    cenarios = definir_cenarios_features()
    cenarios[auditoria.CENARIO_AUDITADO] = cenarios[auditoria.CENARIO_AUDITADO] + [coluna]
    assert auditoria.testar_isolamento_variaveis_alvo(cenarios, *golds)["sucesso"] is False


def test_consumo_do_dia_alvo_em_x_e_detectado_nas_previsoes(golds):
    colunas = [c for c in definir_cenarios_features()[auditoria.CENARIO_AUDITADO] if c in golds[0].columns] + ["consumo"]
    assert auditoria.testar_invariancia_adversarial_modelos(golds[0], ["2026-03-06"], colunas)["sucesso"] is False


def test_media_com_dado_futuro_e_detectada_nas_previsoes(golds):
    def media_com_shift(gold):
        return gold.assign(media_5d=gold["consumo"].shift(-1))
    assert auditoria.testar_invariancia_adversarial_modelos(golds[0], ["2026-03-06"], preparar=media_com_shift)["sucesso"] is False


def test_emissao_com_dias_de_enchente_e_avaliada(golds):
    rel = auditoria.testar_invariancia_adversarial_modelos(golds[0], ["2026-09-04"])
    assert rel["sucesso"] is True, rel["erros"]
