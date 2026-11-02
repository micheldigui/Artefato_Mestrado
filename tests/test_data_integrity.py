import pytest
import pandas as pd
from pathlib import Path
from src.core.config import GOLD, SILVER, RESULTADOS


def test_gold_tables_integrity():
    gold_t1_path = GOLD / "gold_t1.parquet"
    gold_t2_path = GOLD / "gold_t2.parquet"
    
    assert gold_t1_path.exists(), "gold_t1.parquet deve existir"
    assert gold_t2_path.exists(), "gold_t2.parquet deve existir"
    
    df_t1 = pd.read_parquet(gold_t1_path)
    df_t2 = pd.read_parquet(gold_t2_path)
    
    assert len(df_t1) > 0, "gold_t1 não pode estar vazio"
    assert len(df_t2) > 0, "gold_t2 não pode estar vazio"
    
    # Validação de horizontes: t2 não pode ter plano humano
    assert "plano_nutricionista" not in df_t2.columns, "gold_t2 não pode conter plano_nutricionista"
    assert "reserva_discente" not in df_t2.columns, "gold_t2 não pode conter reserva_discente"
    
    # Datas de emissão devem ser sextas-feiras (dayofweek == 4)
    assert (df_t1["origem"].dt.dayofweek == 4).all(), "Todas as origens em T+1 devem ser sextas-feiras"
    assert (df_t2["origem"].dt.dayofweek == 4).all(), "Todas as origens em T+2 devem ser sextas-feiras"


def test_dissertacao_artifacts_exist():
    pasta_tabelas = RESULTADOS / "tabelas_dissertacao"
    tabelas_esperadas = [
        "tabela_03_desempenho_cenarios_retrospectivo.csv",
        "tabela_04_modelos_vs_nutricionista_t1_retrospectivo.csv",
        "tabela_05_politicas_margens_retrospectivo.csv",
        "tabela_06_modelos_t2_compras_retrospectivo.csv",
        "tabela_07_desempenho_prospectivo_set_out.csv",
        "tabela_08_resiliencia_climatica_setembro_enchentes.csv",
        "tabela_10_sintese_comparativa_retrospectivo_vs_prospectivo.csv",
        "tabela_fronteira_modelo_hibrido_retrospectivo.csv",
        "tabela_08b_previsoes_diarias_enchentes.csv",
        "tabela_09_margens_empiricas_quantis.csv",
        "tabela_11_cenario_3_clima_congelado.csv",
        "tabela_12_comparacao_bootstrap_semanal.csv"
    ]
    for tab in tabelas_esperadas:
        p = pasta_tabelas / tab
        assert p.exists(), f"Tabela esperada {tab} não encontrada!"
        assert p.stat().st_size > 0, f"Tabela {tab} está vazia!"

    pasta_figuras = RESULTADOS / "figuras" / "dissertacao"
    figuras_esperadas = [
        "figura_18_ablation_cenarios_horizontes_retrospectivo.png",
        "figura_19_confronto_taxonomia_ensembles_retrospectivo.png",
        "figura_20_modelos_vs_nutricionista_t1_retrospectivo.png",
        "figura_21_fronteira_equilibrio_modelo_hibrido_retrospectivo.png",
        "figura_22_margens_seguranca_conformal_retrospectivo.png",
        "figura_23_tradeoff_sobras_deficits_custo_retrospectivo.png",
        "figura_24_serie_temporal_bimestre_set_out_chuva_vs_consumo.png",
        "figura_25_confronto_diario_setembro_enchentes_ia_vs_nutri.png",
        "figura_26_validacao_outubro_estabilidade.png",
        "figura_27_balanco_sobras_faltas_prospectivo.png",
        "figura_28_sintese_retrospectivo_vs_prospectivo.png"
    ]
    for fig in figuras_esperadas:
        p = pasta_figuras / fig
        assert p.exists(), f"Figura esperada {fig} não encontrada!"
        assert p.stat().st_size > 1000, f"Figura {fig} parece corrompida (tamanho muito pequeno)!"


def test_sintese_usa_todos_os_dias_e_recorte_declarado():
    sintese = pd.read_csv(RESULTADOS / "tabelas_dissertacao" / "tabela_10_sintese_comparativa_retrospectivo_vs_prospectivo.csv", sep=";")
    assert sintese["N_Dias"].tolist() == [123, 33, 28, 5]


def test_previsoes_congeladas_incluem_dias_com_evento_registrado():
    congelado = pd.read_parquet(RESULTADOS / "modelos" / "predicoes_congelado_set_out.parquet")
    assert len(congelado) == 33
    assert int(congelado["evento_registrado"].sum()) == 5
