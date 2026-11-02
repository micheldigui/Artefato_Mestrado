from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.app import pipeline as p
from src.app.modelos import preparar_fato
from src.app.storage import carregar_fato_diaria_completa
from src.core.config import RESULTADOS
from src.core.gold_features import construir_emissao
from src.core.models import definir_cenarios_features
from src.core.selecao import carregar_selecao, treinar_congelado


def previsao_do_torneio(data):
    preds = pd.read_parquet(RESULTADOS / "modelos" / "predicoes_detalhadas_torneio.parquet")
    ref = carregar_selecao()["t1"]
    filtro = (preds["horizonte"] == "t1") & (preds["cenario"] == ref["cenario"]) & (preds["algoritmo"] == ref["algoritmo"])
    return float(preds[filtro & (preds["data"] == data)]["previsao"].iloc[0])


def test_simulador_reproduz_o_walk_forward_da_pesquisa():
    sim = p.simular_previsao_data("2026-03-10", modelo_escolhido=p.MODELO_IA)
    assert sim["pred_ia"] == previsao_do_torneio("2026-03-10")
    assert sim["regime_modelo"] == "Walk-forward até 06/03/2026"


def test_simulador_reproduz_o_modelo_congelado_da_pesquisa():
    congelado = pd.read_parquet(RESULTADOS / "modelos" / "predicoes_congelado_set_out.parquet").set_index("data")
    sim = p.simular_previsao_data("2026-09-15")
    assert sim["pred_ia"] == congelado.loc["2026-09-15", "previsao_ia"]
    assert sim["pred_base"] == congelado.loc["2026-09-15", "previsao_hibrido"]


def test_simulador_nunca_usa_clima_observado():
    sim = p.simular_previsao_data("2026-09-10", cenario_escolhido="Cenário 3 = Cenário 2 + Clima")
    assert sim["fonte_clima"] == "Previsão Arquivada na Emissão"
    assert sim["chuva_utilizada"] == pytest.approx(19.6, abs=0.1)


def test_simulador_abre_com_o_campeao_e_oferece_todos_os_modelos():
    assert p.opcoes_simulador("consumo")[0] == p.MODELO_HIBRIDO
    assert p.opcoes_simulador("compras")[0] == p.NOME_COMPRAS
    assert len(p.opcoes_simulador("consumo")) == 22
    assert len(p.opcoes_simulador("compras")) == 20
    assert not any("Híbrido" in m or "Manual" in m for m in p.opcoes_simulador("compras"))
    assert p.rotulo_simulador("compras", p.NOME_COMPRAS).endswith("(campeão)")


def test_algoritmo_do_simulador_reproduz_o_torneio():
    preds = pd.read_parquet(RESULTADOS / "modelos" / "predicoes_detalhadas_torneio.parquet")
    filtro = (preds["horizonte"] == "t1") & (preds["cenario"] == "Cenario_2_Com_Cardapio") & (preds["algoritmo"] == "Lasso")
    esperado = float(preds[filtro & (preds["data"] == "2026-03-10")]["previsao"].iloc[0])
    assert p.simular_previsao_data("2026-03-10", modelo_escolhido="Lasso")["pred_ia"] == esperado


def test_margem_do_simulador_e_a_do_proprio_algoritmo():
    assert p.margem_do_modelo(p.MARGEM_PADRAO, "ia", 0.0, "t1", 2, "Lasso") > 0
    assert p.margem_do_modelo(p.MARGEM_PADRAO, "hibrido", 0.0, "t1", 2, None) == p.calcular_margem(p.MARGEM_PADRAO, "hibrido")


def test_descricao_da_margem_nas_compras_usa_a_margem_do_extratrees():
    assert "+36 ref" in p.descricao_margem("compras", p.MARGEM_PADRAO)
    assert "+21 ref" in p.descricao_margem("consumo", p.MARGEM_PADRAO)
    assert "Cobertura" in p.descricao_margem("compras", p.MARGEM_SEM)


def test_compras_usam_o_modelo_congelado_escolhido():
    ref = carregar_selecao()["t2"]
    gold = p.carregar_gold("t2")
    colunas = [c for c in definir_cenarios_features()[ref["cenario"]] if c in gold.columns]
    linha = gold[gold["data"] == "2026-09-15"]
    relativa = np.clip(treinar_congelado(gold, ref["algoritmo"], colunas).predict(linha[colunas]), 0.2, 2.5)
    sim = p.simular_previsao_data("2026-09-15", objetivo="compras", cenario_escolhido=p.CENARIO_COMPRAS, modelo_escolhido=p.MODELO_IA)
    assert sim["pred_ia"] == float(np.floor(relativa[0] * linha["nivel_base"].iloc[0] + 0.5))


def test_t2_nao_usa_quantitativo_nem_com_hibrido():
    df, _ = p.executar_pipeline_previsao(horizonte_alvo="Pos_Semana_t2", modelo_escolhido=p.MODELO_HIBRIDO)
    assert df["estimativa_nutri"].isna().all()
    assert (df["pred_base_combinada"] == df["ia_pura"]).all()
    sim = p.simular_previsao_data("2026-03-10", objetivo="compras", modelo_escolhido=p.MODELO_HIBRIDO)
    assert sim["pred_base"] == sim["pred_ia"]


def test_hibrido_sem_quantitativo_usa_apenas_a_ia():
    sim = p.simular_previsao_data("2026-12-02")
    assert sim["nutri_utilizado"] is None
    assert sim["pred_base"] == sim["pred_ia"]


def test_dias_atipicos_nao_entram_nas_medias_do_app():
    fato = preparar_fato(carregar_fato_diaria_completa())
    linha = construir_emissao(fato, pd.Timestamp("2026-09-18"), "t1")
    for coluna in ("minimo_5d", "ultimo_consumo", "mesmo_dia_ha_1_ocorrencias"):
        assert not linha[coluna].isin([84, 103, 138, 14, 23]).any(), coluna


def test_emissao_operacional_igual_a_gold_com_fato_original():
    fato = preparar_fato(carregar_fato_diaria_completa())
    gold = pd.read_parquet(Path(__file__).parent.parent / "dados" / "gold" / "gold_t1.parquet")
    origem = pd.Timestamp("2026-08-28")
    emissao = construir_emissao(fato, origem, "t1").sort_values("data").reset_index(drop=True)
    arquivo = gold[gold["origem"] == origem].sort_values("data").reset_index(drop=True)
    for coluna in ("nivel_base", "media_5d", "efeito_proteina", "efeito_evento", "dias_ate_fim_bloco"):
        pd.testing.assert_series_equal(emissao[coluna].astype(float), arquivo[coluna].astype(float), check_names=False)


def test_app_nao_depende_da_base_antiga_nem_de_senhas_no_codigo():
    raiz = Path(__file__).parent.parent / "src" / "app"
    assert "gold_dataset_ml_limpo" not in (raiz / "pipeline.py").read_text(encoding="utf-8")
    principal = (raiz / "main.py").read_text(encoding="utf-8")
    assert "2026\"" not in principal[principal.index("SENHA_NUTRI ="):principal.index("def senha_confere")]
    assert "senha_rapida in" not in principal
