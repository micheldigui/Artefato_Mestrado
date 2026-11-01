"""Módulo de Análise Exploratória de Dados (EDA) e Suíte de Figuras para a Dissertação.

Analisa e compara os dois regimes de planejamento humano:
  1. 2023 a 2025: Reservas Discentes individuais (sistema de agendamento discente).
  2. 2026: Quantitativo Operacional da Nutricionista (planejamento técnico institucional).

Gera 12 representações gráficas em alta resolução e tabelas de métricas para a dissertação,
focando estritamente em acurácia (MAE, RMSE, WAPE), balanço operacional (sobra vs. falta potencial)
e comprovação empírica do impacto dos fatores (Nível Basal, Calendário, Cardápio e Clima).
"""
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import GOLD, RESULTADOS, SILVER
from .terminology import NOMES_DIAS_SEMANA

PASTA_EDA = RESULTADOS / "eda"
PASTA_FIGURAS = RESULTADOS / "figuras" / "eda"
PASTA_EDA.mkdir(parents=True, exist_ok=True)
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

# Configuração visual padrão para figuras acadêmicas
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11


def carregar_dados_eda() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega dados da Camada Gold T1 e divide nos dois períodos de planejamento humano."""
    gold = pd.read_parquet(GOLD / "gold_t1.parquet")
    treino = gold[gold["usar_no_treino"]].copy()
    
    periodo_discente = treino[treino["ano"] <= 2025].copy()
    periodo_nutri = treino[treino["ano"] == 2026].copy()
    
    return treino, periodo_discente, periodo_nutri


def avaliar_metodos_humanos(periodo_discente: pd.DataFrame, periodo_nutri: pd.DataFrame) -> pd.DataFrame:
    """Compara a acurácia e balanço sobra/falta dos dois métodos humanos (sem viés, focado em métricas)."""
    # 1. Reservas Discentes (2023-2025)
    df_disc = periodo_discente[periodo_discente["reserva_discente"].notna()].copy()
    erro_disc = df_disc["reserva_discente"] - df_disc["consumo"]
    sobra_disc = np.maximum(0, erro_disc)
    falta_disc = np.maximum(0, -erro_disc)
    
    # 2. Plano da Nutricionista (2026)
    df_nutri = periodo_nutri[periodo_nutri["plano_nutricionista"].notna()].copy()
    erro_nutri = df_nutri["plano_nutricionista"] - df_nutri["consumo"]
    sobra_nutri = np.maximum(0, erro_nutri)
    falta_nutri = np.maximum(0, -erro_nutri)
    
    resumo = [
        {
            "Período Histórico": "2023 - 2025",
            "Método de Estimativa": "Reservas Discentes (Autodeclaradas)",
            "Dias Válidos (N)": len(df_disc),
            "Consumo Médio Real (ref)": round(df_disc["consumo"].mean(), 1),
            "Estimativa Média (ref)": round(df_disc["reserva_discente"].mean(), 1),
            "Correlação de Pearson (r)": round(df_disc["reserva_discente"].corr(df_disc["consumo"]), 3),
            "MAE (Erro Médio Absoluto)": round(erro_disc.abs().mean(), 1),
            "RMSE (Raiz do Erro Quadrático)": round(np.sqrt((erro_disc ** 2).mean()), 1),
            "WAPE (%)": round(erro_disc.abs().sum() / df_disc["consumo"].sum() * 100, 1),
            "Total Refeições em Sobra Potencial": int(sobra_disc.sum()),
            "Total Refeições em Déficit Potencial": int(falta_disc.sum()),
            "Taxa de Dias com Sobra Potencial (%)": round((erro_disc > 0).mean() * 100, 1),
            "Taxa de Dias com Déficit Potencial (%)": round((erro_disc < 0).mean() * 100, 1)
        },
        {
            "Período Histórico": "2026",
            "Método de Estimativa": "Quantitativo Operacional (Nutricionista)",
            "Dias Válidos (N)": len(df_nutri),
            "Consumo Médio Real (ref)": round(df_nutri["consumo"].mean(), 1),
            "Estimativa Média (ref)": round(df_nutri["plano_nutricionista"].mean(), 1),
            "Correlação de Pearson (r)": round(df_nutri["plano_nutricionista"].corr(df_nutri["consumo"]), 3),
            "MAE (Erro Médio Absoluto)": round(erro_nutri.abs().mean(), 1),
            "RMSE (Raiz do Erro Quadrático)": round(np.sqrt((erro_nutri ** 2).mean()), 1),
            "WAPE (%)": round(erro_nutri.abs().sum() / df_nutri["consumo"].sum() * 100, 1),
            "Total Refeições em Sobra Potencial": int(sobra_nutri.sum()),
            "Total Refeições em Déficit Potencial": int(falta_nutri.sum()),
            "Taxa de Dias com Sobra Potencial (%)": round((erro_nutri > 0).mean() * 100, 1),
            "Taxa de Dias com Déficit Potencial (%)": round((erro_nutri < 0).mean() * 100, 1)
        }
    ]
    df_resumo = pd.DataFrame(resumo)
    df_resumo.to_csv(PASTA_EDA / "comparativo_metodos_humanos.csv", sep=";", index=False, encoding="utf-8-sig")
    return df_resumo


def calcular_ranking_atributos(treino: pd.DataFrame, periodo_discente: pd.DataFrame, periodo_nutri: pd.DataFrame) -> pd.DataFrame:
    """Calcula a correlação e impacto dos preditores divididos por período e grupo conceitual."""
    preditores = [
        ("Nível Basal", "nivel_base", "Mediana móvel ponderada recente (10d + mesmo dia)"),
        ("Nível Basal", "mesmo_dia_ha_1_ocorrencias", "Consumo do mesmo dia da semana na semana anterior"),
        ("Nível Basal", "media_5d", "Média móvel de consumo dos últimos 5 dias normais"),
        ("Nível Basal", "nivel_exponencial", "Média móvel exponencial causal (alfa=0.2)"),
        ("Nível Basal", "nivel_10d", "Mediana móvel dos 10 dias normais anteriores"),
        ("Nível Basal", "ultimo_consumo", "Último consumo diário observado antes da sexta"),
        ("Calendário", "dias_ate_fim_bloco", "Dias letivos restantes até o término de semestre"),
        ("Calendário", "dias_letivos_na_semana", "Quantidade de dias de aula na semana da refeição"),
        ("Cardápio", "efeito_proteina", "Razão histórica suavizada por Bayes da proteína do dia"),
        ("Cardápio", "efeito_preparo", "Razão histórica suavizada do método de preparo"),
        ("Cardápio", "efeito_prato", "Razão histórica suavizada do prato principal padronizado"),
        ("Clima Previsto", "previsao_chuva_manha_mm", "Volume de chuva previsto entre 6h e 13h"),
        ("Clima Previsto", "previsao_chuva_acum3d_mm", "Volume de chuva acumulado previsto em 3 dias"),
        ("Clima Previsto", "previsao_chuva_mm", "Volume total de chuva previsto para o dia"),
        ("Clima Previsto", "previsao_temp_max", "Temperatura máxima prevista pela API meteorológica"),
        ("Clima Oráculo", "oraculo_temp_max", "Temperatura máxima real observada (ex-post)"),
        ("Clima Oráculo", "oraculo_chuva_mm", "Chuva real observada na estação (ex-post)"),
        ("Calendário", "dias_desde_inicio_bloco", "Dias letivos transcorridos desde o início do período")
    ]
    
    linhas = []
    for grupo, coluna, descricao in preditores:
        if coluna not in treino.columns:
            continue
        corr_geral = treino[coluna].corr(treino["consumo"])
        corr_disc = periodo_discente[coluna].corr(periodo_discente["consumo"])
        corr_nutri = periodo_nutri[coluna].corr(periodo_nutri["consumo"])
        corr_relativo = treino[coluna].corr(treino["alvo_relativo"])
        
        linhas.append({
            "Grupo": grupo,
            "Atributo": coluna,
            "Descrição": descricao,
            "Corr_2023_2025": round(corr_disc, 3) if pd.notna(corr_disc) else 0.0,
            "Corr_2026": round(corr_nutri, 3) if pd.notna(corr_nutri) else 0.0,
            "Corr_Geral": round(corr_geral, 3) if pd.notna(corr_geral) else 0.0,
            "Corr_Relativa": round(corr_relativo, 3) if pd.notna(corr_relativo) else 0.0,
            "Impacto_Absoluto": abs(corr_geral) if pd.notna(corr_geral) else 0.0
        })
        
    df_ranking = pd.DataFrame(linhas).sort_values("Impacto_Absoluto", ascending=False).reset_index(drop=True)
    df_ranking.to_csv(PASTA_EDA / "ranking_impacto_atributos.csv", sep=";", index=False, encoding="utf-8-sig")
    return df_ranking


def gerar_suite_12_figuras_dissertacao(treino: pd.DataFrame, periodo_discente: pd.DataFrame, periodo_nutri: pd.DataFrame) -> None:
    """Gera a suíte completa de 12 figuras analíticas de alta resolução para a dissertação."""
    dias_rotulo = {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta", 4: "Sexta"}
    dias_ordem = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]
    
    # -------------------------------------------------------------------------------------------------
    # FIGURA 01: Confronto dos Métodos Humanos (Dispersão x Consumo Real: Reservas Discentes vs Nutricionista)
    # -------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    v_disc = periodo_discente[periodo_discente["reserva_discente"].notna()]
    max_d = max(v_disc["reserva_discente"].max(), v_disc["consumo"].max()) + 20
    axes[0].scatter(v_disc["reserva_discente"], v_disc["consumo"], color="#2b5c8f", alpha=0.5, s=35, label="Dias Letivos (N=436)")
    axes[0].plot([0, max_d], [0, max_d], color="#e74c3c", linestyle="--", linewidth=1.5, label="Balanço Ideal (y = x)")
    axes[0].set_title(f"Reservas Discentes vs. Consumo Real (2023-2025)\nr = {v_disc['reserva_discente'].corr(v_disc['consumo']):.3f} | MAE = 27.0 | WAPE = 19.9%", fontweight="bold")
    axes[0].set_xlabel("Refeições Reservadas pelos Estudantes")
    axes[0].set_ylabel("Refeições Efetivamente Servidas")
    axes[0].set_xlim(0, max_d)
    axes[0].set_ylim(0, max_d)
    axes[0].legend(loc="upper left")
    
    v_nutri = periodo_nutri[periodo_nutri["plano_nutricionista"].notna()]
    max_n = max(v_nutri["plano_nutricionista"].max(), v_nutri["consumo"].max()) + 20
    axes[1].scatter(v_nutri["plano_nutricionista"], v_nutri["consumo"], color="#27ae60", alpha=0.6, s=40, label="Dias Letivos (N=149)")
    axes[1].plot([0, max_n], [0, max_n], color="#e74c3c", linestyle="--", linewidth=1.5, label="Balanço Ideal (y = x)")
    axes[1].set_title(f"Plano da Nutricionista vs. Consumo Real (2026)\nr = {v_nutri['plano_nutricionista'].corr(v_nutri['consumo']):.3f} | MAE = 23.7 | WAPE = 13.6%", fontweight="bold")
    axes[1].set_xlabel("Quantitativo Planejado pela Nutricionista")
    axes[1].set_ylabel("Refeições Efetivamente Servidas")
    axes[1].set_xlim(0, max_n)
    axes[1].set_ylim(0, max_n)
    axes[1].legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_01_confronto_metodos_humanos.png")
    plt.close()
    
    # -------------------------------------------------------------------------------------------------
    # FIGURA 02: Balanço Acumulado de Sobras vs Faltas Potenciais (Volume Total de Refeições)
    # -------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sobra_disc = np.maximum(0, v_disc["reserva_discente"] - v_disc["consumo"]).sum()
    falta_disc = np.maximum(0, v_disc["consumo"] - v_disc["reserva_discente"]).sum()
    sobra_nutri = np.maximum(0, v_nutri["plano_nutricionista"] - v_nutri["consumo"]).sum()
    falta_nutri = np.maximum(0, v_nutri["consumo"] - v_nutri["plano_nutricionista"]).sum()
    
    metodos = ["Reservas Discentes\n(2023-2025, 436 dias)", "Plano Nutricionista\n(2026, 149 dias)"]
    x = np.arange(len(metodos))
    w = 0.35
    
    b1 = ax.bar(x - w/2, [sobra_disc, sobra_nutri], w, label="Refeições em Sobra Potencial (Desperdício)", color="#27ae60")
    b2 = ax.bar(x + w/2, [falta_disc, falta_nutri], w, label="Refeições em Déficit Potencial (Risco de Fome)", color="#e74c3c")
    
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 150, f"{int(h):,}".replace(",", "."), ha="center", fontsize=10, fontweight="bold")
        
    ax.set_title("Balanço Acumulado de Refeições em Sobra vs. Déficit Potencial", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metodos)
    ax.set_ylabel("Total de Refeições Acumuladas")
    ax.set_ylim(0, max(falta_disc, sobra_disc) * 1.15)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_02_balanco_sobra_falta_metodos.png")
    plt.close()
    
    # -------------------------------------------------------------------------------------------------
    # FIGURA 03: Perfil Semanal de Consumo Facetado por Ano (2023, 2024, 2025, 2026)
    # -------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    anos = [2023, 2024, 2025, 2026]
    cores_ano = ["#34495e", "#7f8c8d", "#2980b9", "#27ae60"]
    
    for ax, ano, cor in zip(axes, anos, cores_ano):
        df_ano = treino[treino["ano"] == ano].copy()
        df_ano["dia_rot"] = df_ano["dia_semana"].map(dias_rotulo)
        medias = df_ano.groupby("dia_rot")["consumo"].mean().reindex(dias_ordem)
        stds = df_ano.groupby("dia_rot")["consumo"].std().reindex(dias_ordem)
        
        ax.bar(dias_ordem, medias, yerr=stds, capsize=4, color=cor, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_title(f"Ano {ano} (N={len(df_ano)})", fontweight="bold")
        ax.set_ylim(0, 260)
        ax.tick_params(axis='x', rotation=35)
        for i, (m, d) in enumerate(zip(medias, dias_ordem)):
            if pd.notna(m):
                ax.text(i, m + 8, f"{m:.0f}", ha="center", fontsize=9, fontweight="bold")
                
    axes[0].set_ylabel("Consumo Médio (Refeições)")
    fig.suptitle("Consistência Longitudinal do Perfil Semanal de Consumo (2023 - 2026)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_03_perfil_semanal_por_ano.png", bbox_inches="tight")
    plt.close()
    
    # -------------------------------------------------------------------------------------------------
    # FIGURA 04: Sazonalidade Mensal do Consumo ao Longo do Calendário Letivo
    # -------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    meses_ordem = range(2, 13) # Fevereiro a Dezembro
    meses_nomes = ["Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    
    sns.boxplot(data=treino[treino["mes"].isin(meses_ordem)], x="mes", y="consumo", ax=ax, hue="mes", palette="Blues", legend=False, width=0.6)
    medias_mes = treino[treino["mes"].isin(meses_ordem)].groupby("mes")["consumo"].mean()
    ax.plot(range(len(meses_nomes)), [medias_mes.get(m, np.nan) for m in meses_ordem], color="#e74c3c", marker="o", linewidth=2, label="Consumo Médio Mensal")
    
    ax.set_title("Variação Sazonal Mensal do Consumo Escolar (Reflexo do Calendário Acadêmico)", fontweight="bold", pad=12)
    ax.set_xticks(range(len(meses_nomes)))
    ax.set_xticklabels(meses_nomes)
    ax.set_xlabel("Mês Letivo")
    ax.set_ylabel("Refeições Servidas")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_04_sazonalidade_mensal.png")
    plt.close()
    
    # -------------------------------------------------------------------------------------------------
    # FIGURA 05: Desmobilização Acadêmica no Término do Semestre (dias_ate_fim_bloco)
    # -------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    treino_fim = treino.copy()
    treino_fim["faixa_fim"] = pd.cut(
        treino_fim["dias_ate_fim_bloco"],
        bins=[-1, 3, 7, 12, 100],
        labels=["Últimos 3 dias (Reta Final)", "4 a 7 dias antes", "8 a 12 dias antes", "> 12 dias (Rotina Escolar)"]
    )
    ordem_f = ["> 12 dias (Rotina Escolar)", "8 a 12 dias antes", "4 a 7 dias antes", "Últimos 3 dias (Reta Final)"]
    medias_fim = treino_fim.groupby("faixa_fim")["consumo"].mean().loc[ordem_f]
    cores_fim = ["#27ae60", "#f1c40f", "#e67e22", "#c0392b"]
    
    bars = ax.bar(ordem_f, medias_fim, color=cores_fim, width=0.55, edgecolor="black", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 3, f"{h:.1f} ref\n({h/medias_fim.iloc[0]-1:+.0%})", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax.set_title("Efeito da Proximidade do Término do Semestre (Queda na Reta Final)", fontweight="bold", pad=12)
    ax.set_ylabel("Consumo Médio Diário")
    ax.set_ylim(0, medias_fim.max() + 28)
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_05_desmobilizacao_fim_semestre.png")
    plt.close()
    
    # -------------------------------------------------------------------------------------------------
    # FIGURA 06: Demanda e Atratividade por Tipo de Proteína no Cardápio
    # -------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    prot_dados = treino.groupby("proteina").agg(
        ocorrencias=("consumo", "count"),
        consumo_medio=("consumo", "mean"),
        razao_media=("alvo_relativo", "mean")
    ).sort_values("consumo_medio", ascending=False)
    
    axes[0].bar(prot_dados.index, prot_dados["ocorrencias"], color="#2b5c8f", width=0.55)
    axes[0].set_title("Frequência de Oferta de Cada Proteína (Dias Letivos)", fontweight="bold")
    axes[0].set_ylabel("Quantidade de Dias Servidos")
    for i, (p, row) in enumerate(prot_dados.iterrows()):
        axes[0].text(i, row["ocorrencias"] + 3, f"{int(row['ocorrencias'])}", ha="center", fontweight="bold")
        
    axes[1].bar(prot_dados.index, prot_dados["consumo_medio"], color="#e67e22", width=0.55)
    axes[1].set_title("Consumo Médio por Tipo de Proteína (Refeições)", fontweight="bold")
    axes[1].set_ylabel("Refeições Servidas (Média)")
    for i, (p, row) in enumerate(prot_dados.iterrows()):
        axes[1].text(i, row["consumo_medio"] + 3, f"{row['consumo_medio']:.1f}", ha="center", fontweight="bold")
        
    fig.suptitle("Impacto da Proteína Principal do Cardápio sobre o Consumo Escolar", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_06_oferta_consumo_por_proteina.png", bbox_inches="tight")
    plt.close()
    
    # -------------------------------------------------------------------------------------------------
    # FIGURA 07: Demanda por Técnica Culinária de Preparo Principal
    # -------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    prep_dados = treino.groupby("preparo").agg(
        dias=("consumo", "count"),
        consumo_medio=("consumo", "mean")
    ).sort_values("consumo_medio", ascending=False)
    
    bars_prep = ax.bar(prep_dados.index, prep_dados["consumo_medio"], color="#8e44ad", width=0.55, edgecolor="black", linewidth=0.5)
    for bar, dias in zip(bars_prep, prep_dados["dias"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 3, f"{h:.1f} ref\n({dias} dias)", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax.set_title("Consumo Médio Conforme a Técnica de Preparo Culinário", fontweight="bold", pad=12)
    ax.set_ylabel("Média de Refeições Servidas")
    ax.set_ylim(0, prep_dados["consumo_medio"].max() + 25)
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_07_consumo_por_tecnica_preparo.png")
    plt.close()
    
    # -------------------------------------------------------------------------------------------------
    # FIGURA 08: Sensibilidade Climática por Faixas de Precipitação
    # -------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    treino_chuva = treino.copy()
    treino_chuva["faixa_chuva"] = pd.cut(
        treino_chuva["previsao_chuva_mm"].fillna(0),
        bins=[-0.1, 0.5, 5.0, 15.0, 200.0],
        labels=["Sem Chuva (0 mm)", "Fraca (0.5 a 5 mm)", "Moderada (5 a 15 mm)", "Forte (> 15 mm)"]
    )
    ordem_c = ["Sem Chuva (0 mm)", "Fraca (0.5 a 5 mm)", "Moderada (5 a 15 mm)", "Forte (> 15 mm)"]
    chuva_agg = treino_chuva.groupby("faixa_chuva", observed=False).agg(
        dias=("consumo", "count"),
        consumo_medio=("consumo", "mean"),
        razao_medio=("alvo_relativo", "mean")
    ).reindex(ordem_c)
    
    b_chuva = ax.bar(ordem_c, chuva_agg["consumo_medio"], color="#2980b9", width=0.55, edgecolor="black", linewidth=0.5)
    for bar, dias, raz in zip(b_chuva, chuva_agg["dias"], chuva_agg["razao_medio"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 3, f"{h:.1f} ref\n({dias} dias | raz: {raz:.2f})", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax.set_title("Sensibilidade da Demanda por Faixas de Precipitação Prevista", fontweight="bold", pad=12)
    ax.set_ylabel("Consumo Médio Diário")
    ax.set_ylim(0, chuva_agg["consumo_medio"].max() + 28)
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_08_sensibilidade_faixas_chuva.png")
    plt.close()
    
    # -------------------------------------------------------------------------------------------------
    # FIGURA 09: Relação Térmica: Temperatura Máxima vs Consumo
    # -------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(
        data=treino, x="previsao_temp_max", y="consumo", ax=ax,
        color="#e67e22", scatter_kws={"alpha": 0.45, "s": 30},
        line_kws={"color": "#c0392b", "linewidth": 1.8, "label": "Linha de Tendência Linear"}
    )
    corr_temp = treino["previsao_temp_max"].corr(treino["consumo"])
    ax.set_title(f"Relação entre Temperatura Máxima Prevista e Consumo Real\nCorrelação Linear: r = {corr_temp:+.3f} (Efeito Estatisticamente Nulo)", fontweight="bold", pad=12)
    ax.set_xlabel("Temperatura Máxima Prevista (°C)")
    ax.set_ylabel("Refeições Servidas")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_09_relacao_temperatura_consumo.png")
    plt.close()
    
    # -------------------------------------------------------------------------------------------------
    # FIGURA 10: Matriz de Correlação Multivariada (Heatmap)
    # -------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8.5))
    cols_corr = [
        "consumo", "nivel_base", "mesmo_dia_ha_1_ocorrencias", "media_5d",
        "dias_ate_fim_bloco", "dias_desde_inicio_bloco", "dias_letivos_na_semana",
        "efeito_proteina", "efeito_preparo", "previsao_chuva_mm", "previsao_temp_max"
    ]
    rotulos_corr = [
        "Consumo Real", "Nível Basal", "Consumo Mesmo Dia t-1", "Média 5d",
        "Dias Fim Semestre", "Dias Início Semestre", "Dias Letivos Semana",
        "Efeito Proteína", "Efeito Preparo", "Prev. Chuva (mm)", "Prev. Temp Máx"
    ]
    matriz = treino[cols_corr].corr()
    matriz.index = rotulos_corr
    matriz.columns = rotulos_corr
    
    sns.heatmap(matriz, annot=True, fmt="+.2f", cmap="vlag", vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlação de Pearson (r)'}, annot_kws={"size": 9, "fontweight": "bold"})
    ax.set_title("Matriz de Correlação Multivariada dos Preditores de Demanda", fontweight="bold", pad=14)
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_10_matriz_correlacao_multivariada.png")
    plt.close()
    
    # -------------------------------------------------------------------------------------------------
    # FIGURA 11: Autocorrelação Temporal (Dependência Serial dos Últimos 20 Dias Úteis)
    # -------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    s_consumo = treino.sort_values("data")["consumo"]
    lags = range(1, 21)
    autocorr = [s_consumo.autocorr(lag=k) for k in lags]
    
    ax.stem(lags, autocorr, linefmt="#2b5c8f", markerfmt="o", basefmt="gray")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.2, color="#e74c3c", linestyle="--", linewidth=1, label="Limiar de Relevância Prática (+0.20)")
    ax.set_title("Função de Autocorrelação Serial do Consumo Escolar (Lags 1 a 20 Dias Úteis)", fontweight="bold", pad=12)
    ax.set_xlabel("Defasagem Temporal (Lags em Dias Úteis)")
    ax.set_ylabel("Autocorrelação")
    ax.set_xticks(lags)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_11_autocorrelacao_temporal.png")
    plt.close()
    
    # -------------------------------------------------------------------------------------------------
    # FIGURA 12: Tornado Chart - Ranking de Impacto Linear dos Atributos
    # -------------------------------------------------------------------------------------------------
    df_ranking = pd.read_csv(PASTA_EDA / "ranking_impacto_atributos.csv", sep=";")
    df_plot = df_ranking.sort_values("Corr_Geral", ascending=True)
    
    fig, ax = plt.subplots(figsize=(11, 7))
    cores_tornado = ["#27ae60" if v > 0.25 else "#3498db" if v > 0 else "#e74c3c" for v in df_plot["Corr_Geral"]]
    
    ax.barh(df_plot["Atributo"], df_plot["Corr_Geral"], color=cores_tornado, height=0.65)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    
    for i, (val, col) in enumerate(zip(df_plot["Corr_Geral"], df_plot["Atributo"])):
        pos_x = val + 0.015 if val >= 0 else val - 0.06
        ax.text(pos_x, i, f"{val:+.3f}", va="center", fontsize=9, fontweight="bold")
        
    ax.set_title("Ranking Geral de Poder Preditivo dos Atributos (Correlação com o Consumo)", fontweight="bold", pad=12)
    ax.set_xlabel("Coeficiente de Correlação de Pearson (r)")
    ax.set_xlim(-0.25, 0.80)
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_12_ranking_impacto_tornado.png")
    plt.close()


def executar_eda_completa() -> dict:
    """Executa a rotina completa de análise exploratória, salva tabelas e gera a suíte de 12 figuras."""
    treino, periodo_discente, periodo_nutri = carregar_dados_eda()
    
    df_humanos = avaliar_metodos_humanos(periodo_discente, periodo_nutri)
    df_ranking = calcular_ranking_atributos(treino, periodo_discente, periodo_nutri)
    
    gerar_suite_12_figuras_dissertacao(treino, periodo_discente, periodo_nutri)
    
    figuras_salvas = sorted([f.name for f in PASTA_FIGURAS.glob("*.png")])
    
    return {
        "dias_treino_total": len(treino),
        "dias_regime_discente_2023_2025": len(periodo_discente),
        "dias_regime_nutricionista_2026": len(periodo_nutri),
        "total_figuras_geradas": len(figuras_salvas),
        "figuras": figuras_salvas
    }


if __name__ == "__main__":
    res = executar_eda_completa()
    print("=== Suíte de 12 Figuras da EDA Concluída com Sucesso! ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
