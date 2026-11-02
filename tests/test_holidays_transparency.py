import pytest
from src.app.storage import carregar_fato_diaria_completa
from src.app.pipeline import executar_pipeline_previsao, simular_previsao_data

def test_holiday_finados_excluded_and_zero():
    # 1. Base fato diária
    df_fato = carregar_fato_diaria_completa()
    row_finados = df_fato[df_fato["data_iso"] == "2026-11-02"]
    assert not row_finados.empty, "Data 2026-11-02 deve existir na base fato"
    assert row_finados.iloc[0]["is_dia_letivo"] == False, "02/11/2026 (Finados) deve ser dia não letivo"

    # 2. Pipeline t+1
    df_t1, df_feat_t1 = executar_pipeline_previsao(horizonte_alvo="Semana_t1")
    assert not df_t1.empty, "Pipeline t+1 deve retornar previsões"
    row_t1_finados = df_t1[df_t1["data_iso"] == "2026-11-02"]
    if not row_t1_finados.empty:
        r = row_t1_finados.iloc[0]
        assert r["is_dia_letivo"] == False, "Finados não deve ser letivo no pipeline"
        assert r["recomendacao_final"] == 0, "Finados deve recomendar exatamente 0 refeições"
        assert r["ia_pura"] == 0, "IA pura deve ser 0 para dia não letivo"
        assert r["margem_aplicada"] == 0, "Margem deve ser 0 para dia não letivo"

    # 3. Simulador de Cenários
    sim = simular_previsao_data("2026-11-02")
    assert sim["is_dia_letivo"] == False, "Simulador deve reconhecer dia não letivo"
    assert sim["pred_final"] == 0, "Simulação para feriado deve resultar em 0 refeições"
    assert sim["pred_ia"] == 0, "Previsão bruta da IA deve ser 0 para feriado"
    assert "Feriado" in sim["prato_utilizado"] or "Fechado" in sim["prato_utilizado"], "Prato deve indicar feriado/fechado"


def test_transparency_columns_and_neutral_menu():
    df_t1, _ = executar_pipeline_previsao(horizonte_alvo="Semana_t1")
    
    # Validação de colunas de transparência e auditoria de insumos
    colunas_transparencia = ["status_calendario", "status_cardapio", "status_clima", "status_nutri"]
    for col in colunas_transparencia:
        assert col in df_t1.columns, f"Coluna de proveniência '{col}' deve existir em df_t1"

    # Dia letivo da semana seguinte (03/11) sem cardápio cadastrado na base deve explicitar perfil neutro
    row_terca = df_t1[df_t1["data_iso"] == "2026-11-03"]
    if not row_terca.empty:
        r_terca = row_terca.iloc[0]
        assert "Padrão Neutro" in r_terca["status_cardapio"], "Cardápio não inserido deve ser explicitado como Padrão Neutro"

    # Horizonte t+2 (Compras quinzenais)
    df_t2, _ = executar_pipeline_previsao(horizonte_alvo="Pos_Semana_t2")
    for col in colunas_transparencia:
        assert col in df_t2.columns, f"Coluna de proveniência '{col}' deve existir em df_t2"
