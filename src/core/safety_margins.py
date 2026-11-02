"""Módulo de Margens de Segurança, Modelo Híbrido e Avaliação Retrospectiva vs. Prospectiva.

Implementa a metodologia de apoio à decisão para a dissertação:
  1. Períodos de 2026:
     - Retrospectivo: Fevereiro a Agosto (walk-forward com retreino semanal).
     - Prospectivo: Setembro e Outubro (modelos escolhidos em Fev-Ago, congelados em 31/08, sem retreino).
  2. Recortes de avaliação: o resultado principal usa todos os dias com consumo; o recorte
     secundário retira apenas dias com evento registrado (status_consumo == "atipico"),
     registro que só existe depois do consumo e por isso é declarado como definido depois do fato.
  3. Modelo Híbrido (combinação convexa IA + Nutricionista em T+1).
  4. Margens empíricas por quantil, calibradas em Fev-Ago e aplicadas sem recalibrar em Set-Out.
  5. Comparação pareada entre métodos com bootstrap por semana de emissão.
  6. Balanço Operacional e Custo Financeiro Potencial de Sobras (R$ 14,00 / refeição).
  7. Geração automatizada de Tabelas CSV e Figuras de Alta Resolução para a dissertação.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import GOLD, SILVER, RESULTADOS, SEMENTE_ALEATORIA
from .selecao import (
    FIM_TREINO_CONGELADO, PESO_IA_HIBRIDO, carregar_selecao, combinar_hibrido, executar_selecao, nome_modelo, treinar_congelado
)

PASTA_TABELAS = RESULTADOS / "tabelas_dissertacao"
PASTA_TABELAS.mkdir(parents=True, exist_ok=True)

PASTA_FIGURAS = RESULTADOS / "figuras" / "dissertacao"
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

ARQUIVO_CONGELADO = RESULTADOS / "modelos" / "predicoes_congelado_set_out.parquet"

PRECO_POR_REFEICAO = 14.00

CENARIO_CLIMA = "Cenario_3_Com_Clima"
POLITICA_IA = "IA Pura"
POLITICA_HIBRIDO = "Modelo Híbrido 50/50"

QUANTIS_MARGEM = {"P75": 0.75, "P85": 0.85, "P90": 0.90}
N_BOOTSTRAP = 10000

# Datas de alagamento definidas depois do fato; coincidem com os dias de status_consumo == "atipico" de 2026.
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


def calcular_margens_por_quantil(y_real: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Deriva as margens como quantis empíricos da falta (consumo - previsão), arredondados para cima."""
    residuos = np.asarray(y_real, dtype=float) - np.asarray(y_pred, dtype=float)
    return {nome: float(max(0.0, np.ceil(np.quantile(residuos, q)))) for nome, q in QUANTIS_MARGEM.items()}


def calcular_cobertura(y_real: np.ndarray, y_pred: np.ndarray, margem: float) -> float:
    """Percentual de dias em que previsão + margem atende todo o consumo observado."""
    return round(float(np.mean(np.asarray(y_real, dtype=float) <= np.asarray(y_pred, dtype=float) + margem) * 100.0), 1)


def comparar_por_bootstrap_semanal(semanas: pd.Series, y_real: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Bootstrap pareado por semana de emissão para a diferença de MAE (B - A); negativo favorece B."""
    dif = np.abs(np.asarray(pred_b, dtype=float) - y_real) - np.abs(np.asarray(pred_a, dtype=float) - y_real)
    grupos = [dif[semanas.values == s] for s in pd.unique(semanas)]
    gerador = np.random.default_rng(SEMENTE_ALEATORIA)
    medias = [
        np.concatenate([grupos[i] for i in gerador.integers(0, len(grupos), len(grupos))]).mean()
        for _ in range(N_BOOTSTRAP)
    ]
    inf, sup = np.percentile(medias, [2.5, 97.5])
    return {
        "Diferenca_MAE": round(float(dif.mean()), 2),
        "IC95_Inferior": round(float(inf), 2),
        "IC95_Superior": round(float(sup), 2),
        "Semanas": len(grupos),
        "Conclusao": "Sem diferença estatisticamente clara" if inf <= 0 <= sup else "Diferença estatisticamente clara"
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

    cols_contexto = ["data", "plano_nutricionista", "reserva_discente", "chuva_mm", "temp_max", "status_consumo"]
    cols_presentes = [c for c in cols_contexto if c in silver.columns]

    df_completo = pd.merge(df_preds, silver[cols_presentes], on="data", how="left")

    # Define categorias temporais
    df_completo["periodo"] = "Outro"
    df_completo.loc[(df_completo["data"] >= "2026-02-01") & (df_completo["data"] <= "2026-08-31"), "periodo"] = "Retrospectivo"
    df_completo.loc[(df_completo["data"] >= "2026-09-01") & (df_completo["data"] <= "2026-10-31"), "periodo"] = "Prospectivo"

    df_completo["data_str"] = df_completo["data"].dt.strftime("%Y-%m-%d")
    df_completo["eh_crise_climatica"] = df_completo["data_str"].isin(DIAS_ENCHENTE_SETEMBRO)
    df_completo["evento_registrado"] = df_completo["status_consumo"].eq("atipico")

    return df_completo, silver


def selecionar_referencia(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra as previsões walk-forward do modelo escolhido para o preparo."""
    ref = carregar_selecao()["t1"]
    filtro = (df["horizonte"] == "t1") & (df["cenario"] == ref["cenario"]) & (df["algoritmo"] == ref["algoritmo"])
    return df[filtro].sort_values("data")


# =====================================================================
# 3. Análise Retrospectiva: Fev a Ago de 2026
# =====================================================================

def processar_retrospectivo(df_completo: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
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
    pred_ia_puro = selecionar_referencia(df_retro)
    pred_nutri = pred_ia_puro["plano_nutricionista"].values
    pred_ia = pred_ia_puro["previsao"].values
    y_real = pred_ia_puro["consumo_real"].values

    pred_hibrido_50 = combinar_hibrido(pred_ia, pred_nutri)

    # Pesquisa de Fronteira do Híbrido (w de 0 a 1)
    linhas_fronteira = []
    for w in np.linspace(0.0, 1.0, 21):
        p_h = combinar_hibrido(pred_ia, pred_nutri, w)
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

    # Margens empíricas calibradas somente em Fev-Ago
    margens = {
        POLITICA_IA: calcular_margens_por_quantil(y_real, pred_ia),
        POLITICA_HIBRIDO: calcular_margens_por_quantil(y_real, pred_hibrido_50)
    }

    linhas_politicas = [{
        "Politica": "Nutricionista Institucional", "Quantil": "Sem margem", "Margem_ref": 0.0, "Margem": "+0 ref",
        **calcular_metricas_completas(y_real, pred_nutri)
    }]
    for politica, pred in ((POLITICA_IA, pred_ia), (POLITICA_HIBRIDO, pred_hibrido_50)):
        niveis = [("Sem margem", 0.0)] + list(margens[politica].items())
        for quantil, margem in niveis:
            linhas_politicas.append({
                "Politica": politica, "Quantil": quantil, "Margem_ref": margem, "Margem": f"+{margem:.0f} ref",
                **calcular_metricas_completas(y_real, pred + margem)
            })
    df_politicas_retro = pd.DataFrame(linhas_politicas)

    # Salvar tabelas retrospectivas
    df_modelos_retro.to_csv(PASTA_TABELAS / "tabela_03_desempenho_cenarios_retrospectivo.csv", sep=";", index=False, encoding="utf-8-sig")
    df_comp_t1.to_csv(PASTA_TABELAS / "tabela_04_modelos_vs_nutricionista_t1_retrospectivo.csv", sep=";", index=False, encoding="utf-8-sig")
    df_t2_retro.to_csv(PASTA_TABELAS / "tabela_06_modelos_t2_compras_retrospectivo.csv", sep=";", index=False, encoding="utf-8-sig")
    df_politicas_retro.to_csv(PASTA_TABELAS / "tabela_05_politicas_margens_retrospectivo.csv", sep=";", index=False, encoding="utf-8-sig")
    df_fronteira_hibrido.to_csv(PASTA_TABELAS / "tabela_fronteira_modelo_hibrido_retrospectivo.csv", sep=";", index=False, encoding="utf-8-sig")

    return df_modelos_retro, df_comp_t1, df_t2_retro, df_politicas_retro, margens


# =====================================================================
# 4. Análise Prospectiva: Setembro e Outubro de 2026
# =====================================================================

def prever_com_modelo_congelado(gold: pd.DataFrame, colunas: List[str], algoritmo: str) -> pd.DataFrame:
    """Congela o modelo escolhido com dados até 31/08/2026 e prevê todos os dias com consumo de Set-Out.

    Os atributos da semana usam o consumo anterior a cada sexta, mas não há retreino depois de 31/08.
    """
    from .models import LIMITES_ALVO, arredondar_refeicoes

    modelo = treinar_congelado(gold, algoritmo, colunas)
    teste = gold[(gold["data"] >= FIM_TREINO_CONGELADO) & (gold["data"] <= "2026-10-31") & gold["consumo"].notna()].copy()
    relativa = np.clip(modelo.predict(teste[colunas]), *LIMITES_ALVO)
    teste["previsao_ia"] = arredondar_refeicoes(relativa * teste["nivel_base"].values)
    return teste


def linhas_por_metodo(sub: pd.DataFrame, margens: Dict[str, Dict[str, float]]) -> List[Tuple[str, np.ndarray]]:
    """Lista os métodos avaliados no período prospectivo com o modelo congelado."""
    return [
        ("Nutricionista Institucional", sub["plano_nutricionista"].values),
        ("IA Congelada em 31/08", sub["previsao_ia"].values),
        ("Híbrido 50/50 Congelado", sub["previsao_hibrido"].values),
        ("Híbrido Congelado + Margem P85", sub["previsao_hibrido"].values + margens[POLITICA_HIBRIDO]["P85"])
    ]


def processar_prospectivo(df_completo: pd.DataFrame, margens: Dict[str, Dict[str, float]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Gera tabelas do período prospectivo (Set-Out/2026) com o modelo congelado como resultado principal."""
    from .models import definir_cenarios_features

    df_prosp = df_completo[df_completo["periodo"] == "Prospectivo"].copy()
    gold_t1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    gold_t1["rel_plano_humano"] = (gold_t1["plano_nutricionista"].fillna(gold_t1["reserva_discente"]) / gold_t1["nivel_base"]).fillna(1.0)
    cenarios = definir_cenarios_features()
    ref = carregar_selecao()["t1"]
    cols = [c for c in cenarios[ref["cenario"]] if c in gold_t1.columns]

    silver = pd.read_parquet(SILVER / "silver_fato_diaria.parquet")
    datas_evento = set(silver.loc[silver["status_consumo"].eq("atipico"), "data"])

    te = prever_com_modelo_congelado(gold_t1, cols, ref["algoritmo"])
    te["previsao_hibrido"] = combinar_hibrido(te["previsao_ia"].values, te["plano_nutricionista"].values)
    te["evento_registrado"] = te["data"].isin(datas_evento)
    te["eh_crise_climatica"] = te["data"].dt.strftime("%Y-%m-%d").isin(DIAS_ENCHENTE_SETEMBRO)

    # Cenário 3: mesma especificação congelada, acrescida da previsão de clima arquivada
    cols_clima = [c for c in cenarios[CENARIO_CLIMA] if c in gold_t1.columns]
    te["previsao_ia_com_clima"] = prever_com_modelo_congelado(gold_t1, cols_clima, ref["algoritmo"])["previsao_ia"].values

    colunas_salvas = [
        "data", "origem", "consumo", "plano_nutricionista", "previsao_ia", "previsao_hibrido", "previsao_ia_com_clima",
        "evento_registrado", "eh_crise_climatica", "previsao_chuva_mm", "oraculo_chuva_mm"
    ]
    te[colunas_salvas].to_parquet(ARQUIVO_CONGELADO, index=False)

    recortes = [
        (f"Todos os dias (N={len(te)})", te),
        (f"Sem evento registrado (N={int((~te['evento_registrado']).sum())})", te[~te["evento_registrado"]]),
        (f"Dias de enchente (N={int(te['eh_crise_climatica'].sum())})", te[te["eh_crise_climatica"]])
    ]

    # 1. Resultado principal: modelo congelado, seguido do walk-forward como sensibilidade
    linhas_prosp = []
    for recorte, sub in recortes:
        for metodo, pred in linhas_por_metodo(sub, margens):
            linhas_prosp.append({
                "Configuracao": metodo,
                "Tipo": "Referência humana" if metodo.startswith("Nutricionista") else "Modelo congelado",
                "Regime": "Resultado principal",
                "Recorte": recorte,
                **calcular_metricas_completas(sub["consumo"].values, pred)
            })

    modelos_foco = list(dict.fromkeys([ref["algoritmo"], "Ensemble_Bagging_ExtraTrees", "Ensemble_Voting_Heterogeneo", "Lasso", "Ensemble_Boosting_Gradient"]))
    wf_t1 = df_prosp[(df_prosp["horizonte"] == "t1") & (df_prosp["cenario"] == ref["cenario"])]
    for recorte, filtro in ((recortes[0][0], wf_t1["data"].notna()), (recortes[1][0], ~wf_t1["evento_registrado"])):
        for alg in modelos_foco:
            g = wf_t1[filtro & (wf_t1["algoritmo"] == alg)].sort_values("data")
            linhas_prosp.append({
                "Configuracao": f"Walk-forward: {alg}",
                "Tipo": "Modelo com retreino semanal",
                "Regime": "Análise de sensibilidade",
                "Recorte": recorte,
                **calcular_metricas_completas(g["consumo_real"].values, g["previsao"].values)
            })
        g = wf_t1[filtro & (wf_t1["algoritmo"] == ref["algoritmo"])].sort_values("data")
        linhas_prosp.append({
            "Configuracao": "Walk-forward: Híbrido 50/50",
            "Tipo": "Modelo com retreino semanal",
            "Regime": "Análise de sensibilidade",
            "Recorte": recorte,
            **calcular_metricas_completas(g["consumo_real"].values, combinar_hibrido(g["previsao"].values, g["plano_nutricionista"].values))
        })
    df_desempenho_prosp = pd.DataFrame(linhas_prosp)

    # 2. Setembro: dias sem evento registrado vs. dias de enchente
    setembro = te[te["data"].dt.month == 9]
    linhas_clima = []
    for recorte, sub in (
        (f"Dias sem evento registrado (N={int((~setembro['evento_registrado']).sum())})", setembro[~setembro["evento_registrado"]]),
        (f"Dias de enchente (N={int(setembro['eh_crise_climatica'].sum())})", setembro[setembro["eh_crise_climatica"]])
    ):
        for metodo, pred in linhas_por_metodo(sub, margens):
            linhas_clima.append({"Recorte": recorte, "Metodo": metodo, **calcular_metricas_completas(sub["consumo"].values, pred)})
    df_clima_setembro = pd.DataFrame(linhas_clima)

    enchentes = te[te["eh_crise_climatica"]][[
        "data", "consumo", "plano_nutricionista", "previsao_ia", "previsao_hibrido", "previsao_ia_com_clima",
        "previsao_chuva_mm", "oraculo_chuva_mm"
    ]].round(1)

    # 3. Cenário 3: a previsão de clima teria ajudado?
    linhas_c3 = []
    for recorte, sub in recortes:
        for metodo, pred in (
            ("Nutricionista Institucional", sub["plano_nutricionista"].values),
            ("IA Congelada - Cenário de referência", sub["previsao_ia"].values),
            ("IA Congelada - Cenário 3", sub["previsao_ia_com_clima"].values)
        ):
            linhas_c3.append({"Recorte": recorte, "Metodo": metodo, **calcular_metricas_completas(sub["consumo"].values, pred)})
    df_cenario_3 = pd.DataFrame(linhas_c3)

    # 4. Cobertura das margens congeladas
    ref_retro = selecionar_referencia(df_completo[df_completo["periodo"] == "Retrospectivo"])
    y_retro = ref_retro["consumo_real"].values
    ia_retro = ref_retro["previsao"].values
    hib_retro = combinar_hibrido(ia_retro, ref_retro["plano_nutricionista"].values)
    sem_evento = te[~te["evento_registrado"]]

    linhas_margens = []
    for politica, pred_r, col in ((POLITICA_IA, ia_retro, "previsao_ia"), (POLITICA_HIBRIDO, hib_retro, "previsao_hibrido")):
        for quantil, margem in margens[politica].items():
            linhas_margens.append({
                "Politica": politica,
                "Quantil": quantil,
                "Margem_ref": margem,
                "Cobertura_FevAgo_pct": calcular_cobertura(y_retro, pred_r, margem),
                "Cobertura_SetOut_Todos_pct": calcular_cobertura(te["consumo"].values, te[col].values, margem),
                "Cobertura_SetOut_Sem_Evento_pct": calcular_cobertura(sem_evento["consumo"].values, sem_evento[col].values, margem)
            })
    df_margens = pd.DataFrame(linhas_margens)

    # 5. Comparação pareada por semana de emissão
    retro_semanal = ref_retro.merge(gold_t1[["data", "origem"]], on="data", how="left")
    ia_r = retro_semanal["previsao"].values
    nutri_r = retro_semanal["plano_nutricionista"].values
    amostras = [
        ("Fev-Ago walk-forward (todos os dias)", retro_semanal["origem"], retro_semanal["consumo_real"].values, ia_r, nutri_r, combinar_hibrido(ia_r, nutri_r)),
        (recortes[0][0] + " - congelado", te["origem"], te["consumo"].values, te["previsao_ia"].values, te["plano_nutricionista"].values, te["previsao_hibrido"].values),
        (recortes[1][0] + " - congelado", sem_evento["origem"], sem_evento["consumo"].values, sem_evento["previsao_ia"].values, sem_evento["plano_nutricionista"].values, sem_evento["previsao_hibrido"].values)
    ]
    linhas_boot = []
    for amostra, semanas, y, ia, nutri, hib in amostras:
        for comparacao, a, b in (("IA menos Nutricionista", nutri, ia), ("Híbrido menos Nutricionista", nutri, hib), ("Híbrido menos IA", ia, hib)):
            linhas_boot.append({"Amostra": amostra, "Comparacao": comparacao, **comparar_por_bootstrap_semanal(semanas, y, a, b)})
    df_bootstrap = pd.DataFrame(linhas_boot)

    # 6. Síntese Comparativa Retrospectivo vs. Prospectivo
    def linha_sintese(periodo: str, natureza: str, y: np.ndarray, nutri: np.ndarray, ia: np.ndarray, hib: np.ndarray) -> dict:
        m_nutri, m_ia, m_hib = (calcular_metricas_completas(y, p) for p in (nutri, ia, hib))
        m_marg = calcular_metricas_completas(y, hib + margens[POLITICA_HIBRIDO]["P85"])
        return {
            "Período": periodo,
            "Natureza": natureza,
            "N_Dias": len(y),
            "Nutricionista_MAE": m_nutri["MAE"],
            "Nutricionista_WAPE": f"{m_nutri['WAPE_pct']}%",
            "IA_MAE": m_ia["MAE"],
            "IA_WAPE": f"{m_ia['WAPE_pct']}%",
            "Hibrido_50_MAE": m_hib["MAE"],
            "Hibrido_50_WAPE": f"{m_hib['WAPE_pct']}%",
            "Margem_P85_Deficit_Dias": f"{m_marg['Taxa_Deficit_pct']}%",
            "Margem_P85_Sobra_Custo_RS": f"R$ {m_marg['Custo_Potencial_Sobras_RS']:,.2f}"
        }

    df_sintese = pd.DataFrame([
        linha_sintese("Retrospectivo (Fev-Ago/2026)", "Walk-forward com retreino semanal", y_retro, ref_retro["plano_nutricionista"].values, ia_retro, hib_retro),
        *[
            linha_sintese(f"Prospectivo (Set-Out/2026) - {recorte}", "Modelo congelado em 31/08/2026, métricas reprocessadas em 02/11/2026",
                          sub["consumo"].values, sub["plano_nutricionista"].values, sub["previsao_ia"].values, sub["previsao_hibrido"].values)
            for recorte, sub in recortes
        ]
    ])

    # Salvar tabelas prospectivas
    df_desempenho_prosp.to_csv(PASTA_TABELAS / "tabela_07_desempenho_prospectivo_set_out.csv", sep=";", index=False, encoding="utf-8-sig")
    df_clima_setembro.to_csv(PASTA_TABELAS / "tabela_08_resiliencia_climatica_setembro_enchentes.csv", sep=";", index=False, encoding="utf-8-sig")
    enchentes.to_csv(PASTA_TABELAS / "tabela_08b_previsoes_diarias_enchentes.csv", sep=";", index=False, encoding="utf-8-sig")
    df_margens.to_csv(PASTA_TABELAS / "tabela_09_margens_empiricas_quantis.csv", sep=";", index=False, encoding="utf-8-sig")
    df_sintese.to_csv(PASTA_TABELAS / "tabela_10_sintese_comparativa_retrospectivo_vs_prospectivo.csv", sep=";", index=False, encoding="utf-8-sig")
    df_cenario_3.to_csv(PASTA_TABELAS / "tabela_11_cenario_3_clima_congelado.csv", sep=";", index=False, encoding="utf-8-sig")
    df_bootstrap.to_csv(PASTA_TABELAS / "tabela_12_comparacao_bootstrap_semanal.csv", sep=";", index=False, encoding="utf-8-sig")

    return df_desempenho_prosp, df_clima_setembro, df_sintese


# =====================================================================
# 5. Gerador da Suíte de Figuras de Alta Resolução para a Dissertação
# =====================================================================

def gerar_figuras_dissertacao(df_completo: pd.DataFrame) -> None:
    """Gera todas as figuras científicas em alta resolução para os capítulos de resultados."""
    print("[*] Gerando Suíte de Figuras Científicas da Dissertação...")
    
    df_retro = df_completo[df_completo["periodo"] == "Retrospectivo"].copy()
    df_congelado = pd.read_parquet(ARQUIVO_CONGELADO).sort_values("data")

    sub_nutri_retro = df_retro[df_retro["horizonte"] == "t1"].drop_duplicates("data")
    mae_nutri_retro = float(np.mean(np.abs(sub_nutri_retro["plano_nutricionista"] - sub_nutri_retro["consumo_real"])))

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
    
    ax.axhline(mae_nutri_retro, color="#d62728", linestyle="--", linewidth=1.5, label="Nutricionista Humano (T+1)")
    
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
    ax.axvline(mae_nutri_retro, color="#d62728", linestyle="--", linewidth=1.5, label=f"Nutricionista Institucional ({mae_nutri_retro:.2f} ref)")
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.2f}", (w, bar.get_y() + bar.get_height() / 2),
                    ha="left", va="center", fontsize=8.5, xytext=(3, 0), textcoords="offset points")
                    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(nomes_tax)
    ax.invert_yaxis()
    ax.set_xlabel("MAE (Refeições)")
    ax.set_title("Figura 19: Desempenho Comparativo da Taxonomia de Ensembles (T+1 - Cenário 2, Fev-Ago/2026)")
    ax.set_xlim(0, 42)
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    fig.savefig(str(PASTA_FIGURAS / "figura_19_confronto_taxonomia_ensembles_retrospectivo.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 20: Modelos Preditivos vs Nutricionista (Dispersão e Resíduos em T+1)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    g_ia_retro = selecionar_referencia(df_retro)
    
    y_r = g_ia_retro["consumo_real"].values
    y_nut = g_ia_retro["plano_nutricionista"].values
    y_m = g_ia_retro["previsao"].values
    
    lims = [min(y_r.min(), y_nut.min(), y_m.min()) - 10, max(y_r.max(), y_nut.max(), y_m.max()) + 10]
    
    ax1.scatter(y_r, y_nut, color="#d62728", alpha=0.6, label="Nutricionista Humano", edgecolors="none")
    ax1.plot(lims, lims, "k--", alpha=0.6)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel("Consumo Real")
    ax1.set_ylabel("Estimativa Humana")
    ax1.set_title(f"Nutricionista Institucional (MAE={np.mean(np.abs(y_nut - y_r)):.2f})")
    ax1.legend(frameon=True)
    
    ax2.scatter(y_r, y_m, color="#1f77b4", alpha=0.6, label="IA", edgecolors="none")
    ax2.plot(lims, lims, "k--", alpha=0.6)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel("Consumo Real")
    ax2.set_ylabel("Predição Machine Learning")
    ax2.set_title(f"IA Pura (MAE={np.mean(np.abs(y_m - y_r)):.2f})")
    ax2.legend(frameon=True)
    
    plt.suptitle("Figura 20: Confronto de Dispersão: Estimativa Humana vs. Machine Learning (Fev-Ago/2026)", fontsize=12)
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
    # FIGURA 22: Margens Empíricas por Quantil e Taxa de Déficit
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    df_pol = pd.read_csv(PASTA_TABELAS / "tabela_05_politicas_margens_retrospectivo.csv", sep=";")
    
    sub_ia = df_pol[df_pol["Politica"] == POLITICA_IA]
    sub_hib = df_pol[df_pol["Politica"] == "Modelo Híbrido 50/50"]
    
    ax1.plot(sub_ia["Margem_ref"], sub_ia["Taxa_Deficit_pct"], marker="o", color="#1f77b4", linewidth=2, label="IA Pura")
    ax1.plot(sub_hib["Margem_ref"], sub_hib["Taxa_Deficit_pct"], marker="s", color="#2ca02c", linewidth=2, label="Híbrido 50/50")
    ax1.axhline(15.0, color="#d62728", linestyle="--", linewidth=1.2, label="Limite de Tolerância (15%)")
    ax1.set_xlabel("Margem de Segurança Adicionada (refeições)")
    ax1.set_ylabel("% Dias com Déficit")
    ax1.set_title("Redução do Risco de Desabastecimento")
    ax1.legend(frameon=True)
    
    ax2.plot(sub_ia["Margem_ref"], sub_ia["Custo_Potencial_Sobras_RS"], marker="o", color="#1f77b4", linewidth=2, label="IA Pura")
    ax2.plot(sub_hib["Margem_ref"], sub_hib["Custo_Potencial_Sobras_RS"], marker="s", color="#2ca02c", linewidth=2, label="Híbrido 50/50")
    ax2.set_xlabel("Margem de Segurança Adicionada (refeições)")
    ax2.set_ylabel("Custo Potencial de Sobras (R$)")
    ax2.set_title("Custo Financeiro Potencial de Desperdício")
    ax2.legend(frameon=True)
    
    plt.suptitle("Figura 22: Margens Empíricas por Quantil Calibradas em Fev-Ago/2026", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(PASTA_FIGURAS / "figura_22_margens_seguranca_conformal_retrospectivo.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 24: Série Temporal Bimestre Setembro-Outubro (Prospectivo)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    
    datas = df_congelado["data"]
    y_real = df_congelado["consumo"].values
    y_nutri = df_congelado["plano_nutricionista"].values
    y_ia = df_congelado["previsao_ia"].values
    y_hib = df_congelado["previsao_hibrido"].values
    chuva = df_congelado["oraculo_chuva_mm"].values
    
    ax1.plot(datas, y_real, "ko-", linewidth=2, markersize=5, label="Consumo Real")
    ax1.plot(datas, y_nutri, "r--", linewidth=1.8, label="Nutricionista Institucional")
    ax1.plot(datas, y_ia, "b-.", linewidth=1.8, label="IA Congelada")
    ax1.plot(datas, y_hib, "g-", linewidth=2, label="Modelo Híbrido 50/50")
    
    # Destacar as datas de enchente extrema
    for d_str in DIAS_ENCHENTE_SETEMBRO:
        dt = pd.to_datetime(d_str)
        if datas.eq(dt).any():
            ax1.axvspan(dt - pd.Timedelta(hours=12), dt + pd.Timedelta(hours=12), color="#d62728", alpha=0.2)
            
    ax1.set_ylabel("Refeições Servidas")
    ax1.set_title(f"Figura 24: Período Prospectivo Setembro–Outubro de 2026 (Modelo Congelado em 31/08, N={len(datas)})")
    ax1.legend(frameon=True, loc="lower center", ncol=4)
    
    # Gráfico de chuva no painel inferior
    ax2.bar(datas, chuva, color="#17becf", edgecolor="black", width=0.6, label="Precipitação observada (mm)")
    ax2.set_ylabel("Chuva (mm)")
    ax2.set_xlabel("Data de Atendimento")
    ax2.legend(frameon=True, loc="upper right")
    
    plt.tight_layout()
    fig.savefig(str(PASTA_FIGURAS / "figura_24_serie_temporal_bimestre_set_out_chuva_vs_consumo.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 25: Setembro: Dias sem Evento Registrado vs. Enchentes
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.8))
    df_clima = pd.read_csv(PASTA_TABELAS / "tabela_08_resiliencia_climatica_setembro_enchentes.csv", sep=";")
    
    sns.barplot(data=df_clima, x="Recorte", y="MAE", hue="Metodo", palette=["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"], ax=ax, edgecolor="black")
    ax.set_title("Figura 25: Erro em Dias sem Evento Registrado vs. Dias de Enchente (Setembro de 2026)")
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
    # FIGURA 26: Desempenho Diário em Outubro
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.8))
    df_out = df_congelado[df_congelado["data"].dt.month == 10]
    
    datas_out = df_out["data"]
    y_real_out = df_out["consumo"].values
    y_nut_out = df_out["plano_nutricionista"].values
    y_ia_out = df_out["previsao_ia"].values
    y_hib_out = df_out["previsao_hibrido"].values
    
    ax.plot(datas_out, y_real_out, "ko-", linewidth=2, markersize=5, label="Consumo Real")
    ax.plot(datas_out, y_nut_out, "r--", linewidth=1.8, label="Nutricionista Institucional")
    ax.plot(datas_out, y_ia_out, "b-.", linewidth=1.8, label="IA Congelada")
    ax.plot(datas_out, y_hib_out, "g-", linewidth=2, label="Modelo Híbrido 50/50")
    
    ax.set_ylabel("Refeições Servidas")
    ax.set_title(f"Figura 26: Desempenho Diário em Outubro de 2026 (Modelo Congelado, N={len(df_out)})")
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
        "IA Congelada em 31/08",
        "Híbrido 50/50 Congelado",
        "Híbrido Congelado + Margem P85"
    ]
    principal = df_prosp_met["Regime"].eq("Resultado principal") & df_prosp_met["Recorte"].str.startswith("Todos os dias")
    sub_p = df_prosp_met[principal & df_prosp_met["Configuracao"].isin(ordem_desejada)].copy()
    sub_p["ordem"] = sub_p["Configuracao"].map({nome: i for i, nome in enumerate(ordem_desejada)})
    sub_p = sub_p.sort_values("ordem").reset_index(drop=True)
    
    labels_b = ["Nutricionista", "IA Congelada", "Híbrido Congelado", "Híbrido + Margem P85"]
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
                    
    ax.set_title(f"Figura 27: Balanço de Sobras e Faltas no Período Prospectivo (N={len(df_congelado)} dias)")
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
    
    sub_ia = df_pol[df_pol["Politica"] == POLITICA_IA]
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
    ax.set_title("Figura 23: Curva de Trade-off: Risco de Desabastecimento vs. Custo de Sobras (Fev-Ago/2026)")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig.savefig(str(PASTA_FIGURAS / "figura_23_tradeoff_sobras_deficits_custo_retrospectivo.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 28: Síntese Visual Comparativa (Retrospectivo vs. Prospectivo)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.8))
    df_sint = pd.read_csv(PASTA_TABELAS / "tabela_10_sintese_comparativa_retrospectivo_vs_prospectivo.csv", sep=";")
    
    cat_labels = [
        f"Retrospectivo\n(Fev-Ago / N={df_sint['N_Dias'].iloc[0]})",
        f"Prospectivo Todos os Dias\n(Set-Out / N={df_sint['N_Dias'].iloc[1]})",
        f"Prospectivo Sem Evento\n(Set-Out / N={df_sint['N_Dias'].iloc[2]})",
        f"Dias de Enchente\n(Setembro / N={df_sint['N_Dias'].iloc[3]})"
    ]
    x = np.arange(len(cat_labels))
    w = 0.25

    nutri_vals = df_sint["Nutricionista_MAE"].values
    ia_vals = df_sint["IA_MAE"].values
    hib_vals = df_sint["Hibrido_50_MAE"].values
    
    r1 = ax.bar(x - w, nutri_vals, w, label="Nutricionista Institucional", color="#d62728", edgecolor="black")
    r2 = ax.bar(x, ia_vals, w, label="IA (Modelo de Referência)", color="#1f77b4", edgecolor="black")
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
    
    _, registro = executar_selecao()
    print(f"\n0. Modelos escolhidos em Fev-Ago: {registro['vencedores']}")
    print("\n1. Processando Período Retrospectivo (Fevereiro a Agosto de 2026)...")
    m_retro, c_t1, t2_retro, pol_retro, margens = processar_retrospectivo(df_completo)
    print(f"   -> {len(m_retro)} configurações avaliadas.")
    print(f"   -> Top 3 Modelos T+1: {c_t1['Modelo / Origem'].iloc[1:4].tolist()}")
    
    print("\n2. Processando Período Prospectivo (Setembro e Outubro de 2026 - Modelo Congelado)...")
    d_prosp, c_set, sintese = processar_prospectivo(df_completo, margens)
    print(f"   -> Avaliação com todos os dias, sem evento registrado e dias de enchente calculada.")
    
    print("\n3. Gerando Suíte de Figuras Científicas da Dissertação...")
    gerar_figuras_dissertacao(df_completo)
    
    print("\n" + "=" * 70)
    print("SÍNTESE COMPARATIVA: RETROSPECTIVO VS. PROSPECTIVO")
    print("=" * 70)
    print(sintese.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    executar_pipeline_completo()
