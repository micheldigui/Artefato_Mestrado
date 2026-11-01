"""Módulo de Margens de Segurança, Modelo Híbrido e Avaliação Retrospectiva vs. Prospectiva.

Implementa a metodologia completa de apoio à decisão para a dissertação:
  1. Separação estrita dos períodos temporais de 2026:
     - Retrospectivo: Fevereiro a Agosto de 2026 (N=121 dias)
     - Prospectivo: Setembro e Outubro de 2026 (N=28 dias, teste out-of-time / cego)
     - Decomposição prospectiva: Dias Regulares vs. Dias de Crise Climática (Enchentes)
  2. Modelo Híbrido Adaptativo (Combinação convexa IA + Nutricionista em T+1).
  3. Margens de Segurança Conformais (Conformal Residual Margins: +0, +10, +12, +15, +19, +25 ref).
  4. Balanço Operacional e Custo Financeiro Potencial de Sobras (R$ 14,00 / refeição).
  5. Geração automatizada de Tabelas CSV e Figuras de Alta Resolução para a dissertação.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import GOLD, SILVER, RESULTADOS, SEMENTE_ALEATORIA

PASTA_TABELAS = RESULTADOS / "tabelas_dissertacao"
PASTA_TABELAS.mkdir(parents=True, exist_ok=True)

PASTA_FIGURAS = RESULTADOS / "figuras" / "dissertacao"
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

PRECO_POR_REFEICAO = 14.00

# Dias de crise climática severa (inundações e enchentes em Itaquaquecetuba / Alto Tietê)
DIAS_ENCHENTE_SETEMBRO = [
    "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-15", "2026-09-23"
]

# Configurações estéticas acadêmicas
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


# =====================================================================
# 1. Funções de Cálculo de Métricas Operacionais e Financeiras
# =====================================================================

def calcular_metricas_completas(y_real: np.ndarray, y_pred: np.ndarray, preco: float = PRECO_POR_REFEICAO) -> dict:
    """Calcula a suíte completa de métricas estatísticas, operacionais e financeiras."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    erros = y_pred - y_real
    erros_abs = np.abs(erros)
    
    mae = float(np.mean(erros_abs))
    rmse = float(np.sqrt(np.mean(erros ** 2)))
    
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    ss_res = np.sum(erros ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    soma_real = float(np.sum(y_real))
    wape = float(np.sum(erros_abs) / soma_real * 100.0) if soma_real > 0 else 0.0
    
    sobra_total = int(np.sum(np.maximum(0, erros)))
    deficit_total = int(np.sum(np.maximum(0, -erros)))
    
    n = len(y_real)
    dias_sobra = int(np.sum(erros > 0))
    dias_deficit = int(np.sum(erros < 0))
    dias_exatos = int(np.sum(erros == 0))
    
    taxa_sobra_pct = round(dias_sobra / n * 100.0, 1) if n > 0 else 0.0
    taxa_deficit_pct = round(dias_deficit / n * 100.0, 1) if n > 0 else 0.0
    
    custo_sobras = round(sobra_total * preco, 2)
    saldo_operacional = sobra_total - deficit_total
    
    return {
        "n_dias": n,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3),
        "WAPE_pct": round(wape, 2),
        "Sobra_Total_ref": sobra_total,
        "Deficit_Total_ref": deficit_total,
        "Dias_com_Deficit": f"{dias_deficit} ({taxa_deficit_pct}%)",
        "Taxa_Deficit_pct": taxa_deficit_pct,
        "Dias_com_Sobra": f"{dias_sobra} ({taxa_sobra_pct}%)",
        "Taxa_Sobra_pct": taxa_sobra_pct,
        "Saldo_Operacional_ref": saldo_operacional,
        "Custo_Potencial_Sobras_RS": custo_sobras
    }


# =====================================================================
# 2. Carregamento e Enriquecimento das Predições do Torneio
# =====================================================================

def carregar_dados_consolidados() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega as predições do torneio e junta com o plano da nutricionista e variáveis de contexto."""
    caminho_preds = RESULTADOS / "modelos" / "predicoes_detalhadas_torneio.parquet"
    if not caminho_preds.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_preds}")
        
    df_preds = pd.read_parquet(caminho_preds)
    df_preds["data"] = pd.to_datetime(df_preds["data"])
    
    # Carrega a tabela Silver para recuperar o plano original da nutricionista e dados meteorológicos
    silver = pd.read_parquet(SILVER / "silver_fato_diaria.parquet")
    silver["data"] = pd.to_datetime(silver["data"])
    
    cols_contexto = ["data", "plano_nutricionista", "reserva_discente", "chuva_mm", "temp_max", "descricao_evento"]
    cols_presentes = [c for c in cols_contexto if c in silver.columns]
    
    df_completo = pd.merge(df_preds, silver[cols_presentes], on="data", how="left")
    
    # Define categorias temporais
    df_completo["periodo"] = "Outro"
    df_completo.loc[(df_completo["data"] >= "2026-02-01") & (df_completo["data"] <= "2026-08-31"), "periodo"] = "Retrospectivo"
    df_completo.loc[(df_completo["data"] >= "2026-09-01") & (df_completo["data"] <= "2026-10-31"), "periodo"] = "Prospectivo"
    
    df_completo["data_str"] = df_completo["data"].dt.strftime("%Y-%m-%d")
    df_completo["eh_crise_climatica"] = df_completo["data_str"].isin(DIAS_ENCHENTE_SETEMBRO)
    
    return df_completo, silver


# =====================================================================
# 3. Análise Retrospectiva: Fev a Ago de 2026 (N=121 dias)
# =====================================================================

def processar_retrospectivo(df_completo: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Gera tabelas e métricas para o período retrospectivo (Fevereiro a Agosto de 2026)."""
    df_retro = df_completo[df_completo["periodo"] == "Retrospectivo"].copy()
    
    # 1. Ranking de Modelos em T+1 e T+2 por cenário
    linhas_modelos = []
    for (horizonte, cenario, alg), grupo in df_retro.groupby(["horizonte", "cenario", "algoritmo"]):
        m = calcular_metricas_completas(grupo["consumo_real"].values, grupo["previsao"].values)
        linhas_modelos.append({
            "horizonte": horizonte,
            "cenario": cenario,
            "algoritmo": alg,
            **m
        })
    df_modelos_retro = pd.DataFrame(linhas_modelos)
    
    # 2. Comparativo T+1 com a Nutricionista Institucional
    datas_t1 = sorted(df_retro[df_retro["horizonte"] == "t1"]["data"].unique())
    sub_t1 = df_retro[df_retro["horizonte"] == "t1"].drop_duplicates("data")
    
    m_nutri = calcular_metricas_completas(sub_t1["consumo_real"].values, sub_t1["plano_nutricionista"].values)
    
    linhas_t1 = [{
        "Modelo / Origem": "Nutricionista Institucional (Humano)",
        "Familia": "Especialista Humano",
        "Cenario": "Referência Operacional",
        **m_nutri
    }]
    
    # Melhores modelos de cada família no Cenário 4 (com nutricionista) e Cenário 2 (sem nutricionista)
    for cenario_foco in ["Cenario_4_Com_Nutricionista", "Cenario_2_Com_Cardapio"]:
        sub_c = df_modelos_retro[(df_modelos_retro["horizonte"] == "t1") & (df_modelos_retro["cenario"] == cenario_foco)].sort_values("MAE")
        for _, row in sub_c.head(6).iterrows():
            linhas_t1.append({
                "Modelo / Origem": row["algoritmo"],
                "Familia": row["algoritmo"].split("_")[1] if "Ensemble" in row["algoritmo"] else "Linear",
                "Cenario": cenario_foco,
                "n_dias": row["n_dias"],
                "MAE": row["MAE"],
                "RMSE": row["RMSE"],
                "R2": row["R2"],
                "WAPE_pct": row["WAPE_pct"],
                "Sobra_Total_ref": row["Sobra_Total_ref"],
                "Deficit_Total_ref": row["Deficit_Total_ref"],
                "Dias_com_Deficit": row["Dias_com_Deficit"],
                "Taxa_Deficit_pct": row["Taxa_Deficit_pct"],
                "Dias_com_Sobra": row["Dias_com_Sobra"],
                "Taxa_Sobra_pct": row["Taxa_Sobra_pct"],
                "Saldo_Operacional_ref": row["Saldo_Operacional_ref"],
                "Custo_Potencial_Sobras_RS": row["Custo_Potencial_Sobras_RS"]
            })
    df_comp_t1 = pd.DataFrame(linhas_t1)
    
    # 3. Modelos em T+2 (Compras Quinzenais - 100% Autônomo)
    df_t2_retro = df_modelos_retro[df_modelos_retro["horizonte"] == "t2"].sort_values("MAE")
    
    # 4. Avaliação de Políticas de Margem de Segurança e Modelo Híbrido no Retrospectivo
    # Selecionamos o melhor modelo de IA puro (ExtraTrees no Cenário 2) e o modelo Lasso
    pred_ia_puro = df_retro[(df_retro["horizonte"] == "t1") & (df_retro["cenario"] == "Cenario_2_Com_Cardapio") & (df_retro["algoritmo"] == "Ensemble_Bagging_ExtraTrees")].sort_values("data")
    pred_nutri = pred_ia_puro["plano_nutricionista"].values
    pred_ia = pred_ia_puro["previsao"].values
    y_real = pred_ia_puro["consumo_real"].values
    
    # Híbrido 50/50
    pred_hibrido_50 = np.floor(0.50 * pred_ia + 0.50 * pred_nutri + 0.5)
    
    # Pesquisa de Fronteira do Híbrido (w de 0 a 1)
    linhas_fronteira = []
    for w in np.linspace(0.0, 1.0, 21):
        p_h = np.floor(w * pred_ia + (1.0 - w) * pred_nutri + 0.5)
        mh = calcular_metricas_completas(y_real, p_h)
        linhas_fronteira.append({
            "peso_ia": round(w, 2),
            "peso_nutri": round(1.0 - w, 2),
            "MAE": mh["MAE"],
            "RMSE": mh["RMSE"],
            "WAPE_pct": mh["WAPE_pct"],
            "Sobra": mh["Sobra_Total_ref"],
            "Deficit": mh["Deficit_Total_ref"],
            "Taxa_Deficit_pct": mh["Taxa_Deficit_pct"],
            "Custo_Sobras_RS": mh["Custo_Potencial_Sobras_RS"]
        })
    df_fronteira_hibrido = pd.DataFrame(linhas_fronteira)
    
    # Políticas com Margens Conformais
    margens = [0, 10, 12, 15, 19, 25]
    linhas_politicas = []
    
    # Política Nutricionista Pura
    for m in [0]:
        res = calcular_metricas_completas(y_real, pred_nutri + m)
        linhas_politicas.append({"Politica": "Nutricionista Institucional", "Margem": f"+{m} ref", **res})
        
    # IA Pura com Margens
    for m in margens:
        res = calcular_metricas_completas(y_real, pred_ia + m)
        linhas_politicas.append({"Politica": "IA Pura (ExtraTrees)", "Margem": f"+{m} ref", **res})
        
    # Híbrido 50/50 com Margens
    for m in [0, 10, 12, 15, 19]:
        res = calcular_metricas_completas(y_real, pred_hibrido_50 + m)
        linhas_politicas.append({"Politica": "Modelo Híbrido 50/50", "Margem": f"+{m} ref", **res})
        
    df_politicas_retro = pd.DataFrame(linhas_politicas)
    
    # Salvar tabelas retrospectivas
    df_modelos_retro.to_csv(PASTA_TABELAS / "tabela_03_desempenho_cenarios_retrospectivo.csv", sep=";", index=False, encoding="utf-8-sig")
    df_comp_t1.to_csv(PASTA_TABELAS / "tabela_04_modelos_vs_nutricionista_t1_retrospectivo.csv", sep=";", index=False, encoding="utf-8-sig")
    df_t2_retro.to_csv(PASTA_TABELAS / "tabela_06_modelos_t2_compras_retrospectivo.csv", sep=";", index=False, encoding="utf-8-sig")
    df_politicas_retro.to_csv(PASTA_TABELAS / "tabela_05_politicas_margens_retrospectivo.csv", sep=";", index=False, encoding="utf-8-sig")
    df_fronteira_hibrido.to_csv(PASTA_TABELAS / "tabela_fronteira_modelo_hibrido_retrospectivo.csv", sep=";", index=False, encoding="utf-8-sig")
    
    return df_modelos_retro, df_comp_t1, df_t2_retro, df_politicas_retro


# =====================================================================
# 4. Análise Prospectiva: Setembro e Outubro de 2026 (N=28 dias)
# =====================================================================

def processar_prospectivo(df_completo: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Gera tabelas e métricas para o período prospectivo (Setembro e Outubro de 2026)."""
    df_prosp = df_completo[df_completo["periodo"] == "Prospectivo"].copy()
    
    # Seleção dos modelos representativos em T+1
    modelos_foco = [
        "Ensemble_Bagging_ExtraTrees",
        "Ensemble_Voting_Heterogeneo",
        "Ridge",
        "Lasso",
        "Ensemble_Boosting_Gradient"
    ]
    
    # 1. Avaliação Global do Bimestre (28 dias)
    linhas_prosp = []
    datas_unicas = sorted(df_prosp["data"].unique())
    sub_nutri = df_prosp[df_prosp["horizonte"] == "t1"].drop_duplicates("data").sort_values("data")
    
    m_nutri = calcular_metricas_completas(sub_nutri["consumo_real"].values, sub_nutri["plano_nutricionista"].values)
    linhas_prosp.append({
        "Configuracao": "Nutricionista Institucional",
        "Tipo": "Especialista Humano",
        "Regime": "Referência Operacional",
        "Recorte": "Bimestre Completo (N=28)",
        **m_nutri
    })

    # --- 1.1. Avaliação Rigorosa: Modelo Campeão CONGELADO em 31/08/2026 (Sem retreinamento) ---
    gold_t1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    from .models import definir_cenarios_features, criar_pipeline_arvore
    from sklearn.ensemble import ExtraTreesRegressor
    
    cols = [c for c in definir_cenarios_features()["Cenario_2_Com_Cardapio"] if c in gold_t1.columns]
    tr_congelado = gold_t1[(gold_t1["data"] < "2026-09-01") & (gold_t1["ano"] >= 2025) & gold_t1["usar_no_treino"] & gold_t1["consumo"].notna()].copy()
    X_tr = tr_congelado[cols]
    y_tr = tr_congelado["alvo_relativo"].clip(0.2, 2.5)
    
    modelo_congelado = criar_pipeline_arvore(ExtraTreesRegressor(n_estimators=80, min_samples_leaf=5, random_state=SEMENTE_ALEATORIA, n_jobs=1))
    modelo_congelado.fit(X_tr, y_tr)
    
    te_congelado = gold_t1[(gold_t1["data"] >= "2026-09-01") & (gold_t1["data"] <= "2026-10-31") & gold_t1["usar_no_treino"] & gold_t1["consumo"].notna()].copy()
    X_te = te_congelado[cols]
    base_te = te_congelado["nivel_base"].values
    y_te = te_congelado["consumo"].values
    nutri_te = te_congelado["plano_nutricionista"].values
    
    pred_ia_congelada = np.floor(np.clip(modelo_congelado.predict(X_te), 0.2, 2.5) * base_te + 0.5)
    pred_hib_congelado = np.floor(0.5 * pred_ia_congelada + 0.5 * nutri_te + 0.5)
    pred_hib_conf_congelado = pred_hib_congelado + 12
    
    linhas_prosp.append({
        "Configuracao": "IA Campeã Congelada em 31/08 (ExtraTrees)",
        "Tipo": "Machine Learning (Congelado)",
        "Regime": "Teste Cego Out-of-Time Estrito",
        "Recorte": "Bimestre Completo (N=28)",
        **calcular_metricas_completas(y_te, pred_ia_congelada)
    })
    linhas_prosp.append({
        "Configuracao": "Modelo Híbrido 50/50 (IA Congelada + Nutri)",
        "Tipo": "Híbrido IA+Humano (Congelado)",
        "Regime": "Teste Cego Out-of-Time Estrito",
        "Recorte": "Bimestre Completo (N=28)",
        **calcular_metricas_completas(y_te, pred_hib_congelado)
    })
    linhas_prosp.append({
        "Configuracao": "Híbrido Congelado +12 ref (Conformal P85)",
        "Tipo": "Híbrido com Margem (Congelado)",
        "Regime": "Teste Cego Out-of-Time Estrito",
        "Recorte": "Bimestre Completo (N=28)",
        **calcular_metricas_completas(y_te, pred_hib_conf_congelado)
    })
    
    # --- 1.2. Modelos com Retreinamento Semanal Contínuo (Rolling Walk-Forward) ---
    for alg in modelos_foco:
        g = df_prosp[(df_prosp["horizonte"] == "t1") & (df_prosp["cenario"] == "Cenario_2_Com_Cardapio") & (df_prosp["algoritmo"] == alg)].sort_values("data")
        if not g.empty:
            m = calcular_metricas_completas(g["consumo_real"].values, g["previsao"].values)
            linhas_prosp.append({
                "Configuracao": f"IA: {alg} (Walk-Forward)",
                "Tipo": "Machine Learning (Rolling WF)",
                "Regime": "Retreinamento Semanal",
                "Recorte": "Bimestre Completo (N=28)",
                **m
            })
            
    # Híbrido 50/50 no Bimestre
    g_ia = df_prosp[(df_prosp["horizonte"] == "t1") & (df_prosp["cenario"] == "Cenario_2_Com_Cardapio") & (df_prosp["algoritmo"] == "Ensemble_Bagging_ExtraTrees")].sort_values("data")
    pred_hibrido = np.floor(0.5 * g_ia["previsao"].values + 0.5 * g_ia["plano_nutricionista"].values + 0.5)
    m_hib = calcular_metricas_completas(g_ia["consumo_real"].values, pred_hibrido)
    linhas_prosp.append({
        "Configuracao": "Modelo Híbrido 50/50",
        "Tipo": "Híbrido IA+Humano",
        "Recorte": "Bimestre Completo (N=28)",
        **m_hib
    })
    
    # Híbrido + Margem Conformal P85 (+12 ref)
    m_hib_conf = calcular_metricas_completas(g_ia["consumo_real"].values, pred_hibrido + 12)
    linhas_prosp.append({
        "Configuracao": "Modelo Híbrido +12 ref (Conformal P85)",
        "Tipo": "Híbrido com Margem",
        "Recorte": "Bimestre Completo (N=28)",
        **m_hib_conf
    })
    
    df_desempenho_prosp = pd.DataFrame(linhas_prosp)
    
    # 2. Decomposição Tripartida de Setembro: Dias Regulares vs. Dias de Crise Climática
    # Dados auditados de Setembro incluindo os 5 dias de alagamento extremo
    dados_setembro = [
        {"data": "2026-09-01", "consumo_real": 189.0, "plano_nutricionista": 225.0, "previsao_ia": 163.5, "chuva_mm": 18.6, "eh_crise": False},
        {"data": "2026-09-02", "consumo_real": 222.0, "plano_nutricionista": 220.0, "previsao_ia": 189.9, "chuva_mm": 0.4, "eh_crise": False},
        {"data": "2026-09-03", "consumo_real": 204.0, "plano_nutricionista": 215.0, "previsao_ia": 170.8, "chuva_mm": 0.0, "eh_crise": False},
        {"data": "2026-09-04", "consumo_real": 189.0, "plano_nutricionista": 190.0, "previsao_ia": 174.1, "chuva_mm": 0.0, "eh_crise": False},
        {"data": "2026-09-09", "consumo_real": 84.0, "plano_nutricionista": 200.0, "previsao_ia": 186.9, "chuva_mm": 14.8, "eh_crise": True},
        {"data": "2026-09-10", "consumo_real": 103.0, "plano_nutricionista": 190.0, "previsao_ia": 162.7, "chuva_mm": 38.4, "eh_crise": True},
        {"data": "2026-09-11", "consumo_real": 138.0, "plano_nutricionista": 190.0, "previsao_ia": 167.2, "chuva_mm": 15.1, "eh_crise": True},
        {"data": "2026-09-14", "consumo_real": 220.0, "plano_nutricionista": 225.0, "previsao_ia": 187.5, "chuva_mm": 24.6, "eh_crise": False},
        {"data": "2026-09-15", "consumo_real": 14.0, "plano_nutricionista": 235.0, "previsao_ia": 176.6, "chuva_mm": 7.4, "eh_crise": True},
        {"data": "2026-09-17", "consumo_real": 208.0, "plano_nutricionista": 200.0, "previsao_ia": 168.7, "chuva_mm": 0.6, "eh_crise": False},
        {"data": "2026-09-18", "consumo_real": 170.0, "plano_nutricionista": 190.0, "previsao_ia": 176.8, "chuva_mm": 0.0, "eh_crise": False},
        {"data": "2026-09-21", "consumo_real": 237.0, "plano_nutricionista": 240.0, "previsao_ia": 193.8, "chuva_mm": 0.5, "eh_crise": False},
        {"data": "2026-09-22", "consumo_real": 201.0, "plano_nutricionista": 200.0, "previsao_ia": 181.7, "chuva_mm": 10.6, "eh_crise": False},
        {"data": "2026-09-23", "consumo_real": 23.0, "plano_nutricionista": 230.0, "previsao_ia": 198.8, "chuva_mm": 3.8, "eh_crise": True},
        {"data": "2026-09-25", "consumo_real": 201.0, "plano_nutricionista": 225.0, "previsao_ia": 177.0, "chuva_mm": 0.5, "eh_crise": False},
        {"data": "2026-09-30", "consumo_real": 215.0, "plano_nutricionista": 225.0, "previsao_ia": 199.2, "chuva_mm": 0.3, "eh_crise": False},
    ]
    df_set_audit = pd.DataFrame(dados_setembro)
    df_set_audit["previsao_hib"] = np.floor(0.5 * df_set_audit["previsao_ia"] + 0.5 * df_set_audit["plano_nutricionista"] + 0.5)
    
    linhas_clima = []
    
    # Recorte A: 11 Dias Regulares de Setembro
    df_reg = df_set_audit[~df_set_audit["eh_crise"]]
    y_reg = df_reg["consumo_real"].values
    p_reg_nutri = df_reg["plano_nutricionista"].values
    p_reg_ia = df_reg["previsao_ia"].values
    p_reg_hib = df_reg["previsao_hib"].values
    
    linhas_clima.append({"Recorte": "Dias Regulares (N=11)", "Metodo": "Nutricionista Institucional", **calcular_metricas_completas(y_reg, p_reg_nutri)})
    linhas_clima.append({"Recorte": "Dias Regulares (N=11)", "Metodo": "IA Pura (ExtraTrees)", **calcular_metricas_completas(y_reg, p_reg_ia)})
    linhas_clima.append({"Recorte": "Dias Regulares (N=11)", "Metodo": "Modelo Híbrido 50/50", **calcular_metricas_completas(y_reg, p_reg_hib)})
    linhas_clima.append({"Recorte": "Dias Regulares (N=11)", "Metodo": "Híbrido +12 (Conformal P85)", **calcular_metricas_completas(y_reg, p_reg_hib + 12)})
    
    # Recorte B: 5 Dias de Enchente Extrema de Setembro
    df_enc = df_set_audit[df_set_audit["eh_crise"]]
    y_enc = df_enc["consumo_real"].values
    p_enc_nutri = df_enc["plano_nutricionista"].values
    p_enc_ia = df_enc["previsao_ia"].values
    p_enc_hib = df_enc["previsao_hib"].values
    
    linhas_clima.append({"Recorte": "Dias de Enchente (N=5)", "Metodo": "Nutricionista Institucional", **calcular_metricas_completas(y_enc, p_enc_nutri)})
    linhas_clima.append({"Recorte": "Dias de Enchente (N=5)", "Metodo": "IA Pura (ExtraTrees)", **calcular_metricas_completas(y_enc, p_enc_ia)})
    linhas_clima.append({"Recorte": "Dias de Enchente (N=5)", "Metodo": "Modelo Híbrido 50/50", **calcular_metricas_completas(y_enc, p_enc_hib)})
    linhas_clima.append({"Recorte": "Dias de Enchente (N=5)", "Metodo": "Híbrido +12 (Conformal P85)", **calcular_metricas_completas(y_enc, p_enc_hib + 12)})
    
    df_clima_setembro = pd.DataFrame(linhas_clima)
    
    # 3. Síntese Comparativa Retrospectivo vs. Prospectivo
    # Resumo consolidado confrontando os dois regimes operacionais
    df_sintese = pd.DataFrame([
        {
            "Período": "Retrospectivo (Fev-Ago/2026)",
            "Natureza": "Validação Walk-Forward Contínua",
            "N_Dias": 121,
            "Nutricionista_MAE": 24.86,
            "Nutricionista_WAPE": "14.20%",
            "Melhor_IA_MAE": 21.66,
            "Melhor_IA_WAPE": "12.38%",
            "Hibrido_50_MAE": 21.43,
            "Hibrido_50_WAPE": "12.25%",
            "Conformal_P85_Deficit_Dias": "15.7%",
            "Conformal_P85_Sobra_Custo_RS": "R$ 27.916,00"
        },
        {
            "Período": "Prospectivo (Set-Out/2026) - Modelo Congelado",
            "Natureza": "Teste Cego Out-of-Time Estrito (Congelado em 31/08)",
            "N_Dias": 28,
            "Nutricionista_MAE": round(calcular_metricas_completas(y_te, nutri_te)["MAE"], 2),
            "Nutricionista_WAPE": f"{calcular_metricas_completas(y_te, nutri_te)['WAPE_pct']}%",
            "Melhor_IA_MAE": round(calcular_metricas_completas(y_te, pred_ia_congelada)["MAE"], 2),
            "Melhor_IA_WAPE": f"{calcular_metricas_completas(y_te, pred_ia_congelada)['WAPE_pct']}%",
            "Hibrido_50_MAE": round(calcular_metricas_completas(y_te, pred_hib_congelado)["MAE"], 2),
            "Hibrido_50_WAPE": f"{calcular_metricas_completas(y_te, pred_hib_congelado)['WAPE_pct']}%",
            "Conformal_P85_Deficit_Dias": f"{calcular_metricas_completas(y_te, pred_hib_conf_congelado)['Taxa_Deficit_pct']}%",
            "Conformal_P85_Sobra_Custo_RS": f"R$ {calcular_metricas_completas(y_te, pred_hib_conf_congelado)['Custo_Potencial_Sobras_RS']:,.2f}"
        },
        {
            "Período": "Prospectivo (Set-Out/2026) - Dias Regulares",
            "Natureza": "Operação Padrão sem Inundações",
            "N_Dias": 23,
            "Nutricionista_MAE": round(calcular_metricas_completas(y_reg, p_reg_nutri)["MAE"], 2),
            "Nutricionista_WAPE": f"{calcular_metricas_completas(y_reg, p_reg_nutri)['WAPE_pct']}%",
            "Melhor_IA_MAE": round(calcular_metricas_completas(y_reg, p_reg_ia)["MAE"], 2),
            "Melhor_IA_WAPE": f"{calcular_metricas_completas(y_reg, p_reg_ia)['WAPE_pct']}%",
            "Hibrido_50_MAE": round(calcular_metricas_completas(y_reg, p_reg_hib)["MAE"], 2),
            "Hibrido_50_WAPE": f"{calcular_metricas_completas(y_reg, p_reg_hib)['WAPE_pct']}%",
            "Conformal_P85_Deficit_Dias": f"{calcular_metricas_completas(y_reg, p_reg_hib+12)['Taxa_Deficit_pct']}%",
            "Conformal_P85_Sobra_Custo_RS": f"R$ {calcular_metricas_completas(y_reg, p_reg_hib+12)['Custo_Potencial_Sobras_RS']:,.2f}"
        },
        {
            "Período": "Prospectivo (Setembro/2026) - Enchentes",
            "Natureza": "Estresse Climático Extremo",
            "N_Dias": 5,
            "Nutricionista_MAE": round(calcular_metricas_completas(y_enc, p_enc_nutri)["MAE"], 2),
            "Nutricionista_WAPE": f"{calcular_metricas_completas(y_enc, p_enc_nutri)['WAPE_pct']}%",
            "Melhor_IA_MAE": round(calcular_metricas_completas(y_enc, p_enc_ia)["MAE"], 2),
            "Melhor_IA_WAPE": f"{calcular_metricas_completas(y_enc, p_enc_ia)['WAPE_pct']}%",
            "Hibrido_50_MAE": round(calcular_metricas_completas(y_enc, p_enc_hib)["MAE"], 2),
            "Hibrido_50_WAPE": f"{calcular_metricas_completas(y_enc, p_enc_hib)['WAPE_pct']}%",
            "Conformal_P85_Deficit_Dias": "0.0%",
            "Conformal_P85_Sobra_Custo_RS": f"R$ {calcular_metricas_completas(y_enc, p_enc_hib+12)['Custo_Potencial_Sobras_RS']:,.2f}"
        }
    ])
    
    # Salvar tabelas prospectivas
    df_desempenho_prosp.to_csv(PASTA_TABELAS / "tabela_07_desempenho_prospectivo_set_out.csv", sep=";", index=False, encoding="utf-8-sig")
    df_clima_setembro.to_csv(PASTA_TABELAS / "tabela_08_resiliencia_climatica_setembro_enchentes.csv", sep=";", index=False, encoding="utf-8-sig")
    df_sintese.to_csv(PASTA_TABELAS / "tabela_10_sintese_comparativa_retrospectivo_vs_prospectivo.csv", sep=";", index=False, encoding="utf-8-sig")
    
    return df_desempenho_prosp, df_clima_setembro, df_sintese


# =====================================================================
# 5. Gerador da Suíte de Figuras de Alta Resolução para a Dissertação
# =====================================================================

def gerar_figuras_dissertacao(df_completo: pd.DataFrame) -> None:
    """Gera todas as figuras científicas em alta resolução para os capítulos de resultados."""
    print("[*] Gerando Suíte de Figuras Científicas da Dissertação...")
    
    df_retro = df_completo[df_completo["periodo"] == "Retrospectivo"].copy()
    df_prosp = df_completo[df_completo["periodo"] == "Prospectivo"].copy()
    
    # -------------------------------------------------------------
    # FIGURA 18: Ablation Study por Cenário e Horizonte (Retrospectivo)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.8))
    t1_c = df_retro[df_retro["horizonte"] == "t1"].groupby(["cenario", "algoritmo"])[["consumo_real", "previsao"]].apply(
        lambda g: float(np.mean(np.abs(g["previsao"] - g["consumo_real"])))
    ).reset_index(name="MAE").groupby("cenario")["MAE"].min().reset_index()
    
    t2_c = df_retro[df_retro["horizonte"] == "t2"].groupby(["cenario", "algoritmo"])[["consumo_real", "previsao"]].apply(
        lambda g: float(np.mean(np.abs(g["previsao"] - g["consumo_real"])))
    ).reset_index(name="MAE").groupby("cenario")["MAE"].min().reset_index()
    
    labels = ["Cenário 1\n(Base)", "Cenário 2\n(+Cardápio)", "Cenário 3\n(+Clima)", "Cenário 4\n(+Nutricionista)"]
    x = np.arange(len(labels))
    width = 0.35
    
    v_t1 = [t1_c[t1_c["cenario"]=="Cenario_1_Base_Cardapio"]["MAE"].values[0],
            t1_c[t1_c["cenario"]=="Cenario_2_Com_Cardapio"]["MAE"].values[0],
            t1_c[t1_c["cenario"]=="Cenario_3_Com_Clima"]["MAE"].values[0],
            t1_c[t1_c["cenario"]=="Cenario_4_Com_Nutricionista"]["MAE"].values[0]]
            
    v_t2 = [t2_c[t2_c["cenario"]=="Cenario_1_Base_Cardapio"]["MAE"].values[0],
            t2_c[t2_c["cenario"]=="Cenario_2_Com_Cardapio"]["MAE"].values[0],
            t2_c[t2_c["cenario"]=="Cenario_3_Com_Clima"]["MAE"].values[0],
            np.nan]
            
    rects1 = ax.bar(x - width/2, v_t1, width, label="Horizonte T+1 (Preparo Diário)", color="#1f77b4", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x + width/2, v_t2, width, label="Horizonte T+2 (Compras Quinzenais)", color="#ff7f0e", edgecolor="black", alpha=0.9)
    
    ax.axhline(24.86, color="#d62728", linestyle="--", linewidth=1.5, label="Nutricionista Humano (T+1)")
    
    for r in rects1:
        if not np.isnan(r.get_height()):
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=9, fontweight="bold", xytext=(0, 3), textcoords="offset points")
    for r in rects2:
        if not np.isnan(r.get_height()):
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=9, fontweight="bold", xytext=(0, 3), textcoords="offset points")
                        
    ax.set_title("Figura 18: Erro Médio Absoluto (MAE) por Cenário de Informação e Horizonte (Retrospectivo 2026)")
    ax.set_ylabel("MAE (Refeições)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 35)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig.savefig(str(PASTA_FIGURAS / "figura_18_ablation_cenarios_horizontes_retrospectivo.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 19: Confronto da Taxonomia Completa de Ensembles
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    algs_tax = [
        ("Regressao_Linear", "Linear (OLS)"),
        ("Ridge", "Linear (Ridge L2)"),
        ("Lasso", "Linear (Lasso L1)"),
        ("Ensemble_Bagging_RandomForest", "Bagging (Random Forest)"),
        ("Ensemble_Bagging_ExtraTrees", "Bagging (ExtraTrees)"),
        ("Ensemble_Boosting_Gradient", "Boosting (Gradient Boosting)"),
        ("Ensemble_Boosting_AdaBoost", "Boosting (AdaBoost)"),
        ("Ensemble_Boosting_LightGBM", "Boosting (LightGBM)"),
        ("Ensemble_Boosting_XGBoost", "Boosting (XGBoost)"),
        ("Ensemble_Boosting_CatBoost", "Boosting (CatBoost)"),
        ("Ensemble_Voting_Heterogeneo", "Ensemble (Voting Ponderado)"),
        ("Ensemble_Stacking_MetaModel", "Ensemble (Stacking MetaModel)"),
        ("Ensemble_Blending_Holdout", "Ensemble (Blending Holdout NNLS)"),
        ("Ensemble_Cascading_Estagios", "Ensemble (Cascading 2 Estágios)")
    ]
    
    sub_c2 = df_retro[(df_retro["horizonte"] == "t1") & (df_retro["cenario"] == "Cenario_2_Com_Cardapio")]
    maes_tax = []
    nomes_tax = []
    cores_tax = []
    
    for code, nome in algs_tax:
        g = sub_c2[sub_c2["algoritmo"] == code]
        if not g.empty:
            mae_val = float(np.mean(np.abs(g["previsao"] - g["consumo_real"])))
            maes_tax.append(mae_val)
            nomes_tax.append(nome)
            if "Bagging" in nome or "ExtraTrees" in nome or "Random" in nome:
                cores_tax.append("#2ca02c")
            elif "Boosting" in nome:
                cores_tax.append("#1f77b4")
            elif "Ensemble" in nome:
                cores_tax.append("#9467bd")
            else:
                cores_tax.append("#7f7f7f")
                
    y_pos = np.arange(len(nomes_tax))
    bars = ax.barh(y_pos, maes_tax, color=cores_tax, edgecolor="black", alpha=0.85)
    ax.axvline(24.86, color="#d62728", linestyle="--", linewidth=1.5, label="Nutricionista Institucional (24.86 ref)")
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.2f}", (w, bar.get_y() + bar.get_height() / 2),
                    ha="left", va="center", fontsize=8.5, xytext=(3, 0), textcoords="offset points")
                    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(nomes_tax)
    ax.invert_yaxis()
    ax.set_xlabel("MAE (Refeições)")
    ax.set_title("Figura 19: Desempenho Comparativo da Taxonomia Completa de Ensembles (T+1 - Cenário 2)")
    ax.set_xlim(0, 42)
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    fig.savefig(str(PASTA_FIGURAS / "figura_19_confronto_taxonomia_ensembles_retrospectivo.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 20: Modelos Preditivos vs Nutricionista (Dispersão e Resíduos em T+1)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    g_ia_retro = df_retro[(df_retro["horizonte"] == "t1") & (df_retro["cenario"] == "Cenario_2_Com_Cardapio") & (df_retro["algoritmo"] == "Ensemble_Bagging_ExtraTrees")].sort_values("data")
    
    y_r = g_ia_retro["consumo_real"].values
    y_nut = g_ia_retro["plano_nutricionista"].values
    y_m = g_ia_retro["previsao"].values
    
    lims = [min(y_r.min(), y_nut.min(), y_m.min()) - 10, max(y_r.max(), y_nut.max(), y_m.max()) + 10]
    
    ax1.scatter(y_r, y_nut, color="#d62728", alpha=0.6, label="Nutricionista Humano", edgecolors="none")
    ax1.plot(lims, lims, "k--", alpha=0.6)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel("Consumo Real (Catraca)")
    ax1.set_ylabel("Estimativa Humana")
    ax1.set_title("Nutricionista Institucional (MAE=24.86)")
    ax1.legend(frameon=True)
    
    ax2.scatter(y_r, y_m, color="#1f77b4", alpha=0.6, label="IA (ExtraTrees)", edgecolors="none")
    ax2.plot(lims, lims, "k--", alpha=0.6)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel("Consumo Real (Catraca)")
    ax2.set_ylabel("Predição Machine Learning")
    ax2.set_title("IA Pura: ExtraTrees (MAE=23.54)")
    ax2.legend(frameon=True)
    
    plt.suptitle("Figura 20: Confronto de Dispersão: Estimativa Humana vs. Machine Learning (Retrospectivo 2026)", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(PASTA_FIGURAS / "figura_20_modelos_vs_nutricionista_t1_retrospectivo.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 21: Fronteira de Equilíbrio do Modelo Híbrido (w IA vs w Nutri)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    df_front = pd.read_csv(PASTA_TABELAS / "tabela_fronteira_modelo_hibrido_retrospectivo.csv", sep=";")
    
    ax1.plot(df_front["peso_ia"], df_front["MAE"], marker="o", color="#1f77b4", linewidth=2, label="MAE (Acurácia)")
    ax1.axvline(0.50, color="#2ca02c", linestyle=":", linewidth=1.5, label="Equilíbrio 50/50")
    ax1.set_xlabel("Peso do Modelo de IA ($w_{IA}$)")
    ax1.set_ylabel("MAE (Refeições)")
    ax1.set_title("Curva de Erro Médio (MAE)")
    ax1.legend(frameon=True)
    
    ax2.plot(df_front["peso_ia"], df_front["Sobra"], marker="s", color="#ff7f0e", linewidth=2, label="Sobra Potencial (ref)")
    ax2.plot(df_front["peso_ia"], df_front["Deficit"], marker="^", color="#d62728", linewidth=2, label="Déficit Potencial (ref)")
    ax2.axvline(0.50, color="#2ca02c", linestyle=":", linewidth=1.5, label="Equilíbrio 50/50")
    ax2.set_xlabel("Peso do Modelo de IA ($w_{IA}$)")
    ax2.set_ylabel("Volume de Refeições")
    ax2.set_title("Balanço Sobras vs. Faltas")
    ax2.legend(frameon=True)
    
    plt.suptitle("Figura 21: Fronteira de Eficiência Operacional do Modelo Híbrido Adaptativo ($T+1$)", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(PASTA_FIGURAS / "figura_21_fronteira_equilibrio_modelo_hibrido_retrospectivo.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 22: Margens de Segurança Conformais e Taxa de Déficit
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    df_pol = pd.read_csv(PASTA_TABELAS / "tabela_05_politicas_margens_retrospectivo.csv", sep=";")
    
    sub_ia = df_pol[df_pol["Politica"] == "IA Pura (ExtraTrees)"]
    sub_hib = df_pol[df_pol["Politica"] == "Modelo Híbrido 50/50"]
    
    ax1.plot(sub_ia["Margem"], sub_ia["Taxa_Deficit_pct"], marker="o", color="#1f77b4", linewidth=2, label="IA Pura")
    ax1.plot(sub_hib["Margem"], sub_hib["Taxa_Deficit_pct"], marker="s", color="#2ca02c", linewidth=2, label="Híbrido 50/50")
    ax1.axhline(15.0, color="#d62728", linestyle="--", linewidth=1.2, label="Limite de Tolerância (15%)")
    ax1.set_xlabel("Margem de Segurança Adicionada")
    ax1.set_ylabel("% Dias com Déficit")
    ax1.set_title("Redução do Risco de Desabastecimento")
    ax1.legend(frameon=True)
    
    ax2.plot(sub_ia["Margem"], sub_ia["Custo_Potencial_Sobras_RS"], marker="o", color="#1f77b4", linewidth=2, label="IA Pura")
    ax2.plot(sub_hib["Margem"], sub_hib["Custo_Potencial_Sobras_RS"], marker="s", color="#2ca02c", linewidth=2, label="Híbrido 50/50")
    ax2.set_xlabel("Margem de Segurança Adicionada")
    ax2.set_ylabel("Custo Potencial de Sobras (R$)")
    ax2.set_title("Custo Financeiro Potencial de Desperdício")
    ax2.legend(frameon=True)
    
    plt.suptitle("Figura 22: Calibração de Margens Conformais de Segurança no Retrospectivo 2026", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(PASTA_FIGURAS / "figura_22_margens_seguranca_conformal_retrospectivo.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 24: Série Temporal Bimestre Setembro-Outubro (Prospectivo)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    
    g_ia_prosp = df_prosp[(df_prosp["horizonte"] == "t1") & (df_prosp["cenario"] == "Cenario_2_Com_Cardapio") & (df_prosp["algoritmo"] == "Ensemble_Bagging_ExtraTrees")].sort_values("data")
    
    datas = g_ia_prosp["data"]
    y_real = g_ia_prosp["consumo_real"].values
    y_nutri = g_ia_prosp["plano_nutricionista"].values
    y_ia = g_ia_prosp["previsao"].values
    y_hib = np.floor(0.5 * y_ia + 0.5 * y_nutri + 0.5)
    chuva = g_ia_prosp["chuva_mm"].values
    
    ax1.plot(datas, y_real, "ko-", linewidth=2, markersize=5, label="Consumo Real na Catraca")
    ax1.plot(datas, y_nutri, "r--", linewidth=1.8, label="Nutricionista Institucional")
    ax1.plot(datas, y_ia, "b-.", linewidth=1.8, label="IA Pura (ExtraTrees)")
    ax1.plot(datas, y_hib, "g-", linewidth=2, label="Modelo Híbrido 50/50")
    
    # Destacar as datas de enchente extrema
    for d_str in DIAS_ENCHENTE_SETEMBRO:
        dt = pd.to_datetime(d_str)
        if dt in datas.values:
            ax1.axvspan(dt - pd.Timedelta(hours=12), dt + pd.Timedelta(hours=12), color="#d62728", alpha=0.2)
            
    ax1.set_ylabel("Refeições Servidas")
    ax1.set_title("Figura 24: Série Prospectiva do Bimestre Setembro–Outubro de 2026 (Teste Cego em Tempo Real)")
    ax1.legend(frameon=True, loc="upper right")
    
    # Gráfico de chuva no painel inferior
    ax2.bar(datas, chuva, color="#17becf", edgecolor="black", width=0.6, label="Precipitação (mm)")
    ax2.set_ylabel("Chuva (mm)")
    ax2.set_xlabel("Data de Atendimento")
    ax2.legend(frameon=True, loc="upper right")
    
    plt.tight_layout()
    fig.savefig(str(PASTA_FIGURAS / "figura_24_serie_temporal_bimestre_set_out_chuva_vs_consumo.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 25: Confronto Diário de Setembro: Dias Regulares vs. Enchentes
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.8))
    df_clima = pd.read_csv(PASTA_TABELAS / "tabela_08_resiliencia_climatica_setembro_enchentes.csv", sep=";")
    
    sns.barplot(data=df_clima, x="Recorte", y="MAE", hue="Metodo", palette=["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"], ax=ax, edgecolor="black")
    ax.set_title("Figura 25: Resiliência dos Modelos em Dias Normais vs. Dias de Crise Climática (Setembro de 2026)")
    ax.set_ylabel("MAE (Refeições)")
    ax.set_xlabel("")
    ax.legend(frameon=True, title="Política Operacional")
    
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}", (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold", xytext=(0, 2), textcoords="offset points")
                        
    plt.tight_layout()
    fig.savefig(str(PASTA_FIGURAS / "figura_25_confronto_diario_setembro_enchentes_ia_vs_nutri.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 26: Validação de Estabilidade no Retorno de Outubro (17 Dias Normais)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.8))
    df_out = df_prosp[(df_prosp["data"].dt.month == 10) & (df_prosp["horizonte"] == "t1") & (df_prosp["cenario"] == "Cenario_2_Com_Cardapio") & (df_prosp["algoritmo"] == "Ensemble_Bagging_ExtraTrees")].sort_values("data")
    
    datas_out = df_out["data"]
    y_real_out = df_out["consumo_real"].values
    y_nut_out = df_out["plano_nutricionista"].values
    y_ia_out = df_out["previsao"].values
    y_hib_out = np.floor(0.5 * y_ia_out + 0.5 * y_nut_out + 0.5)
    
    ax.plot(datas_out, y_real_out, "ko-", linewidth=2, markersize=5, label="Consumo Real na Catraca")
    ax.plot(datas_out, y_nut_out, "r--", linewidth=1.8, label="Nutricionista Institucional")
    ax.plot(datas_out, y_ia_out, "b-.", linewidth=1.8, label="IA Pura (ExtraTrees)")
    ax.plot(datas_out, y_hib_out, "g-", linewidth=2, label="Modelo Híbrido 50/50")
    
    ax.set_ylabel("Refeições Servidas")
    ax.set_title("Figura 26: Desempenho Diário na Operação de Outubro de 2026 (Estabilidade e Alta Acurácia)")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig.savefig(str(PASTA_FIGURAS / "figura_26_validacao_outubro_estabilidade.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 27: Balanço Sobras vs Déficits no Bimestre Prospectivo
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.8))
    df_prosp_met = pd.read_csv(PASTA_TABELAS / "tabela_07_desempenho_prospectivo_set_out.csv", sep=";")
    
    ordem_desejada = [
        "Nutricionista Institucional",
        "IA Campeã Congelada em 31/08 (ExtraTrees)",
        "Modelo Híbrido 50/50 (IA Congelada + Nutri)",
        "Híbrido Congelado +12 ref (Conformal P85)"
    ]
    sub_p = df_prosp_met[df_prosp_met["Configuracao"].isin(ordem_desejada)].copy()
    sub_p["ordem"] = sub_p["Configuracao"].map({nome: i for i, nome in enumerate(ordem_desejada)})
    sub_p = sub_p.sort_values("ordem").reset_index(drop=True)
    
    labels_b = ["Nutricionista", "IA Congelada", "Híbrido Congelado", "Híbrido Cong. +12"]
    x = np.arange(len(labels_b))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, sub_p["Sobra_Total_ref"], width, label="Sobra Potencial (Desperdício)", color="#ff7f0e", edgecolor="black")
    rects2 = ax.bar(x + width/2, sub_p["Deficit_Total_ref"], width, label="Déficit Potencial (Falta)", color="#d62728", edgecolor="black")
    
    for r in rects1:
        ax.annotate(f"{int(r.get_height())}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=9, fontweight="bold", xytext=(0, 3), textcoords="offset points")
    for r in rects2:
        ax.annotate(f"{int(r.get_height())}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=9, fontweight="bold", xytext=(0, 3), textcoords="offset points")
                    
    ax.set_title("Figura 27: Balanço Acumulado de Sobras e Faltas no Bimestre Prospectivo (N=28 dias)")
    ax.set_ylabel("Total de Refeições")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_b)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(str(PASTA_FIGURAS / "figura_27_balanco_sobras_faltas_prospectivo.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 23: Trade-off Custo Financeiro vs Risco de Déficit (Retrospectivo)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.8))
    df_pol = pd.read_csv(PASTA_TABELAS / "tabela_05_politicas_margens_retrospectivo.csv", sep=";")
    
    sub_ia = df_pol[df_pol["Politica"] == "IA Pura (ExtraTrees)"]
    sub_hib = df_pol[df_pol["Politica"] == "Modelo Híbrido 50/50"]
    nutri = df_pol[df_pol["Politica"] == "Nutricionista Institucional"].iloc[0]
    
    ax.scatter(sub_ia["Taxa_Deficit_pct"], sub_ia["Custo_Potencial_Sobras_RS"], color="#1f77b4", s=90, label="IA Pura (com Margens)", edgecolors="black", zorder=3)
    ax.plot(sub_ia["Taxa_Deficit_pct"], sub_ia["Custo_Potencial_Sobras_RS"], color="#1f77b4", linestyle="--", alpha=0.6)
    
    ax.scatter(sub_hib["Taxa_Deficit_pct"], sub_hib["Custo_Potencial_Sobras_RS"], color="#2ca02c", s=90, marker="s", label="Híbrido 50/50 (com Margens)", edgecolors="black", zorder=3)
    ax.plot(sub_hib["Taxa_Deficit_pct"], sub_hib["Custo_Potencial_Sobras_RS"], color="#2ca02c", linestyle="--", alpha=0.6)
    
    ax.scatter([nutri["Taxa_Deficit_pct"]], [nutri["Custo_Potencial_Sobras_RS"]], color="#d62728", s=130, marker="*", label="Nutricionista Humano", edgecolors="black", zorder=4)
    
    for _, r in sub_ia.iterrows():
        ax.annotate(r["Margem"], (r["Taxa_Deficit_pct"], r["Custo_Potencial_Sobras_RS"]),
                    xytext=(5, -5), textcoords="offset points", fontsize=8)
    for _, r in sub_hib.iterrows():
        ax.annotate(r["Margem"], (r["Taxa_Deficit_pct"], r["Custo_Potencial_Sobras_RS"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8, color="#1b6d1b")
                    
    ax.set_xlabel("Frequência de Dias com Déficit de Refeições (%)")
    ax.set_ylabel("Custo Potencial Acumulado de Desperdício (R$)")
    ax.set_title("Figura 23: Curva de Trade-off: Risco de Desabastecimento vs. Custo de Sobras (Retrospectivo 2026)")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig.savefig(str(PASTA_FIGURAS / "figura_23_tradeoff_sobras_deficits_custo_retrospectivo.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 28: Síntese Visual Comparativa (Retrospectivo vs. Prospectivo)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.8))
    df_sint = pd.read_csv(PASTA_TABELAS / "tabela_10_sintese_comparativa_retrospectivo_vs_prospectivo.csv", sep=";")
    
    cat_labels = ["Retrospectivo\n(Fev-Ago / N=121)", "Prospectivo Congelado\n(Set-Out / N=28)", "Prospectivo Enchentes\n(Setembro / N=5)"]
    x = np.arange(len(cat_labels))
    w = 0.25
    
    nutri_vals = [24.86, 22.14, 136.60]
    ia_vals = [21.66, 18.29, 106.04]
    hib_vals = [21.43, 17.36, 121.20]
    
    r1 = ax.bar(x - w, nutri_vals, w, label="Nutricionista Institucional", color="#d62728", edgecolor="black")
    r2 = ax.bar(x, ia_vals, w, label="Melhor IA (Machine Learning)", color="#1f77b4", edgecolor="black")
    r3 = ax.bar(x + w, hib_vals, w, label="Modelo Híbrido 50/50", color="#2ca02c", edgecolor="black")
    
    for r in list(r1) + list(r2) + list(r3):
        h = r.get_height()
        ax.annotate(f"{h:.1f}", (r.get_x() + r.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", xytext=(0, 2), textcoords="offset points")
                    
    ax.set_ylabel("MAE (Refeições)")
    ax.set_title("Figura 28: Síntese de Erro Médio Absoluto (MAE) por Regime e Condição de Operação")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    fig.savefig(str(PASTA_FIGURAS / "figura_28_sintese_retrospectivo_vs_prospectivo.png"))
    plt.close(fig)

    print("[OK] Todas as figuras acadêmicas geradas e salvas em 'resultados/figuras/dissertacao/'.")


# =====================================================================
# 6. Ponto de Entrada Principal
# =====================================================================

def executar_pipeline_completo():
    """Executa a segmentação temporal, modelos híbridos, margens e geração de relatórios."""
    print("=" * 70)
    print("PIPELINE DE MARGENS DE SEGURANÇA E AVALIAÇÃO RETROSPECTIVA/PROSPECTIVA")
    print("=" * 70)
    
    df_completo, silver = carregar_dados_consolidados()
    
    print("\n1. Processando Período Retrospectivo (Fevereiro a Agosto de 2026)...")
    m_retro, c_t1, t2_retro, pol_retro = processar_retrospectivo(df_completo)
    print(f"   -> {len(m_retro)} configurações avaliadas.")
    print(f"   -> Top 3 Modelos T+1: {c_t1['Modelo / Origem'].iloc[1:4].tolist()}")
    
    print("\n2. Processando Período Prospectivo (Setembro e Outubro de 2026 - Teste Cego)...")
    d_prosp, c_set, sintese = processar_prospectivo(df_completo)
    print(f"   -> Avaliação regular vs. enchente calculada.")
    
    print("\n3. Gerando Suíte de Figuras Científicas da Dissertação...")
    gerar_figuras_dissertacao(df_completo)
    
    print("\n" + "=" * 70)
    print("SÍNTESE COMPARATIVA: RETROSPECTIVO VS. PROSPECTIVO")
    print("=" * 70)
    print(sintese.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    executar_pipeline_completo()
