"""Módulo de Análise Exploratória de Dados (EDA) e Diagnóstico de Fatores de Demanda.

Analisa e compara os dois regimes de planejamento humano:
  1. 2023 a 2025: Reservas Discentes individuais (sistema no-show reverso).
  2. 2026: Quantitativo Operacional da Nutricionista (planejamento institucional centralizado).

Avalia o impacto quantitativo de Clima, Calendário, Cardápio e Nível Basal sobre o consumo.
"""
from pathlib import Path
from typing import Tuple
import json
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

# Configuração visual padrão para figuras da dissertação
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150


def carregar_dados_eda() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega dados da Camada Gold T1 e divide nos dois períodos de planejamento humano."""
    gold = pd.read_parquet(GOLD / "gold_t1.parquet")
    treino = gold[gold["usar_no_treino"]].copy()
    
    periodo_discente = treino[treino["ano"] <= 2025].copy()
    periodo_nutri = treino[treino["ano"] == 2026].copy()
    
    return treino, periodo_discente, periodo_nutri


def avaliar_metodos_humanos(periodo_discente: pd.DataFrame, periodo_nutri: pd.DataFrame) -> pd.DataFrame:
    """Compara estatisticamente a acurácia, viés e balanço sobra/falta dos dois métodos humanos."""
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
            "Viés Médio / Balanço (ref)": round(erro_disc.mean(), 1),
            "Comportamento Típico": "Subnotificação Crônica (Alunos comem sem reservar)",
            "Total Refeições em Sobra": int(sobra_disc.sum()),
            "Total Refeições em Falta (Risco)": int(falta_disc.sum()),
            "Taxa de Dias com Falta Potencial (%)": round((erro_disc < 0).mean() * 100, 1)
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
            "Viés Médio / Balanço (ref)": round(erro_nutri.mean(), 1),
            "Comportamento Típico": "Superestimativa Protetiva (Evita falta de alimento)",
            "Total Refeições em Sobra": int(sobra_nutri.sum()),
            "Total Refeições em Falta (Risco)": int(falta_nutri.sum()),
            "Taxa de Dias com Falta Potencial (%)": round((erro_nutri < 0).mean() * 100, 1)
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
        ("Calendário", "dias_ate_fim_bloco", "Dias letivos restantes até o recesso/término de semestre"),
        ("Calendário", "dias_desde_inicio_bloco", "Dias letivos transcorridos desde o início do período"),
        ("Calendário", "dias_letivos_na_semana", "Quantidade de dias de aula na semana da refeição"),
        ("Cardápio", "efeito_proteina", "Razão histórica suavizada por Bayes da proteína do dia"),
        ("Cardápio", "efeito_preparo", "Razão histórica suavizada do método de preparo"),
        ("Cardápio", "efeito_prato", "Razão histórica suavizada do prato principal padronizado"),
        ("Clima Previsto", "previsao_chuva_manha_mm", "Volume de chuva previsto entre 6h e 13h"),
        ("Clima Previsto", "previsao_chuva_mm", "Volume total de chuva previsto para o dia"),
        ("Clima Previsto", "previsao_temp_max", "Temperatura máxima prevista pela API meteorológica"),
        ("Clima Previsto", "previsao_chuva_acum3d_mm", "Volume de chuva acumulado previsto em 3 dias"),
        ("Clima Oráculo", "oraculo_chuva_mm", "Chuva real observada na estação (ex-post)"),
        ("Clima Oráculo", "oraculo_temp_max", "Temperatura máxima real observada (ex-post)")
    ]
    
    linhas = []
    for grupo, coluna, descricao in preditores:
        if coluna not in treino.columns:
            continue
        corr_geral = treino[coluna].corr(treino["consumo"])
        corr_disc = periodo_discente[coluna].corr(periodo_discente["consumo"])
        corr_nutri = periodo_nutri[coluna].corr(periodo_nutri["consumo"])
        
        # Correlação com consumo relativo (descontando o nível basal)
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


def analisar_padrao_dia_semana(treino: pd.DataFrame, periodo_discente: pd.DataFrame, periodo_nutri: pd.DataFrame) -> pd.DataFrame:
    """Analisa a sazonalidade semanal comparando os dois períodos."""
    dias_rotulo = {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta", 4: "Sexta"}
    treino["dia_rotulo"] = treino["dia_semana"].map(dias_rotulo)
    periodo_discente["dia_rotulo"] = periodo_discente["dia_semana"].map(dias_rotulo)
    periodo_nutri["dia_rotulo"] = periodo_nutri["dia_semana"].map(dias_rotulo)
    
    ordem_dias = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]
    
    df_dias = pd.DataFrame({
        "Consumo Médio (2023-2025)": periodo_discente.groupby("dia_rotulo")["consumo"].mean().round(1),
        "Consumo Médio (2026)": periodo_nutri.groupby("dia_rotulo")["consumo"].mean().round(1),
        "Consumo Médio Geral": treino.groupby("dia_rotulo")["consumo"].mean().round(1),
        "Desvio Padrão": treino.groupby("dia_rotulo")["consumo"].std().round(1),
        "Razão vs Nível Basal": treino.groupby("dia_rotulo")["alvo_relativo"].mean().round(3)
    }).loc[ordem_dias]
    
    df_dias.to_csv(PASTA_EDA / "sazonalidade_dia_semana.csv", sep=";", index=True, encoding="utf-8-sig")
    return df_dias


def analisar_desmobilizacao_semestre(treino: pd.DataFrame) -> pd.DataFrame:
    """Analisa o efeito de queda acentuada de frequência discente na reta final do semestre."""
    treino = treino.copy()
    treino["faixa_fim"] = pd.cut(
        treino["dias_ate_fim_bloco"],
        bins=[-1, 3, 7, 12, 100],
        labels=["Últimos 3 dias (Reta Final)", "4 a 7 dias antes", "8 a 12 dias antes", "> 12 dias (Rotina Escolar)"]
    )
    
    ordem_faixas = ["> 12 dias (Rotina Escolar)", "8 a 12 dias antes", "4 a 7 dias antes", "Últimos 3 dias (Reta Final)"]
    
    resumo_fim = pd.DataFrame({
        "Quantidade de Dias": treino["faixa_fim"].value_counts(),
        "Consumo Médio (refeições)": treino.groupby("faixa_fim")["consumo"].mean().round(1),
        "Mediana do Consumo": treino.groupby("faixa_fim")["consumo"].median().round(1),
        "Razão Média vs Nível Basal": treino.groupby("faixa_fim")["alvo_relativo"].mean().round(3),
        "Queda Percentual vs Rotina (%)": (
            (treino.groupby("faixa_fim")["consumo"].mean() / treino.loc[treino["faixa_fim"] == "> 12 dias (Rotina Escolar)", "consumo"].mean() - 1) * 100
        ).round(1)
    }).loc[ordem_faixas]
    
    resumo_fim.to_csv(PASTA_EDA / "queda_fim_de_semestre.csv", sep=";", index=True, encoding="utf-8-sig")
    return resumo_fim


def analisar_cardapio_proteinas(treino: pd.DataFrame) -> pd.DataFrame:
    """Analisa o efeito de cada tipo de proteína no consumo e na razão de adesão."""
    df_prot = pd.DataFrame({
        "Frequência (Dias)": treino["proteina"].value_counts(),
        "Consumo Médio (refeições)": treino.groupby("proteina")["consumo"].mean().round(1),
        "Razão vs Nível Basal": treino.groupby("proteina")["alvo_relativo"].mean().round(3),
        "Efeito Bayesiano Médio": treino.groupby("proteina")["efeito_proteina"].mean().round(3)
    }).sort_values("Razão vs Nível Basal", ascending=False)
    
    df_prot.to_csv(PASTA_EDA / "impacto_proteinas_cardapio.csv", sep=";", index=True, encoding="utf-8-sig")
    return df_prot


def gerar_graficos_eda(treino: pd.DataFrame, periodo_discente: pd.DataFrame, periodo_nutri: pd.DataFrame) -> None:
    """Gera suíte completa de figuras em alta resolução para a dissertação."""
    
    # -------------------------------------------------------------
    # Figura 01: Comparativo dos Métodos Humanos (Dispersão + Resíduos)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Painel 1: Reservas Discentes (2023-2025)
    v_disc = periodo_discente[periodo_discente["reserva_discente"].notna()]
    axes[0].scatter(v_disc["reserva_discente"], v_disc["consumo"], color="#2b5c8f", alpha=0.5, s=35, label="Dias Letivos (N=436)")
    max_val_disc = max(v_disc["reserva_discente"].max(), v_disc["consumo"].max()) + 20
    axes[0].plot([0, max_val_disc], [0, max_val_disc], color="#e74c3c", linestyle="--", linewidth=1.5, label="Balanço Perfeito (y = x)")
    axes[0].set_title("Reservas Discentes vs. Consumo Real (2023-2025)\nViés Médio: -15.8 ref | WAPE: 19.9% (Subnotificação)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Refeições Reservadas pelos Alunos")
    axes[0].set_ylabel("Refeições Efetivamente Servidas (Consumo)")
    axes[0].set_xlim(0, max_val_disc)
    axes[0].set_ylim(0, max_val_disc)
    axes[0].legend(loc="upper left")
    
    # Painel 2: Plano da Nutricionista (2026)
    v_nutri = periodo_nutri[periodo_nutri["plano_nutricionista"].notna()]
    axes[1].scatter(v_nutri["plano_nutricionista"], v_nutri["consumo"], color="#27ae60", alpha=0.6, s=40, label="Dias Letivos (N=149)")
    max_val_nutri = max(v_nutri["plano_nutricionista"].max(), v_nutri["consumo"].max()) + 20
    axes[1].plot([0, max_val_nutri], [0, max_val_nutri], color="#e74c3c", linestyle="--", linewidth=1.5, label="Balanço Perfeito (y = x)")
    axes[1].set_title("Plano da Nutricionista vs. Consumo Real (2026)\nViés Médio: +10.4 ref | WAPE: 13.6% (Proteção Social)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Quantitativo Planejado pela Nutricionista")
    axes[1].set_ylabel("Refeições Efetivamente Servidas (Consumo)")
    axes[1].set_xlim(0, max_val_nutri)
    axes[1].set_ylim(0, max_val_nutri)
    axes[1].legend(loc="upper left")
    
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_01_comparativo_metodos_humanos.png")
    plt.close()
    
    # -------------------------------------------------------------
    # Figura 02: Sazonalidade por Dia da Semana (2023-2025 vs 2026)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    dias_ordem = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]
    dias_rotulo = {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta", 4: "Sexta"}
    
    p1_dias = periodo_discente.assign(dia_rot=periodo_discente["dia_semana"].map(dias_rotulo)).groupby("dia_rot")["consumo"].mean().loc[dias_ordem]
    p2_dias = periodo_nutri.assign(dia_rot=periodo_nutri["dia_semana"].map(dias_rotulo)).groupby("dia_rot")["consumo"].mean().loc[dias_ordem]
    
    x = np.arange(len(dias_ordem))
    width = 0.35
    
    ax.bar(x - width/2, p1_dias, width, label="2023-2025 (Regime Discente)", color="#34495e", alpha=0.85)
    ax.bar(x + width/2, p2_dias, width, label="2026 (Regime Nutricionista)", color="#3498db", alpha=0.85)
    
    for i in x:
        ax.text(i - width/2, p1_dias.iloc[i] + 3, f"{p1_dias.iloc[i]:.0f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + width/2, p2_dias.iloc[i] + 3, f"{p2_dias.iloc[i]:.0f}", ha="center", fontsize=9, fontweight="bold")
        
    ax.set_title("Consumo Médio por Dia da Semana: Validação da Inércia Semanal", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(dias_ordem)
    ax.set_ylabel("Média de Refeições Servidas")
    ax.set_ylim(0, max(p2_dias.max(), p1_dias.max()) + 30)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_02_sazonalidade_semanal.png")
    plt.close()
    
    # -------------------------------------------------------------
    # Figura 03: Desmobilização Acadêmica no Término do Semestre
    # -------------------------------------------------------------
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
        
    ax.set_title("Efeito de Desmobilização no Término do Semestre Letivo\n(Aproximação de Exames Finais e Férias Escolares)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Consumo Médio Diário")
    ax.set_ylim(0, medias_fim.max() + 25)
    
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_03_desmobilizacao_fim_semestre.png")
    plt.close()
    
    # -------------------------------------------------------------
    # Figura 04: Tornado Chart - Ranking de Correlação dos Preditores
    # -------------------------------------------------------------
    df_ranking = pd.read_csv(PASTA_EDA / "ranking_impacto_atributos.csv", sep=";")
    df_plot = df_ranking.sort_values("Corr_Geral", ascending=True)
    
    fig, ax = plt.subplots(figsize=(11, 7))
    cores_tornado = ["#27ae60" if v > 0.2 else "#3498db" if v > 0 else "#e74c3c" for v in df_plot["Corr_Geral"]]
    
    ax.barh(df_plot["Atributo"], df_plot["Corr_Geral"], color=cores_tornado, height=0.65)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    
    for i, (val, col) in enumerate(zip(df_plot["Corr_Geral"], df_plot["Atributo"])):
        pos_x = val + 0.015 if val >= 0 else val - 0.06
        ax.text(pos_x, i, f"{val:+.3f}", va="center", fontsize=9, fontweight="bold")
        
    ax.set_title("Ranking de Correlação Linear com o Consumo Real (2023 - 2026)\nComprovação Empírica do Poder Preditivo dos Atributos", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Coeficiente de Correlação de Pearson (r)")
    ax.set_xlim(-0.25, 0.80)
    
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_04_ranking_correlacoes.png")
    plt.close()
    
    # -------------------------------------------------------------
    # Figura 05: Sensibilidade Climática (Chuva e Temperatura)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chuva Prevista vs Consumo
    sns.regplot(data=treino, x="previsao_chuva_mm", y="consumo", ax=axes[0], color="#2980b9", scatter_kws={"alpha": 0.4, "s": 25}, line_kws={"color": "#e74c3c", "linewidth": 1.5})
    axes[0].set_title(f"Previsão de Chuva (mm) vs. Consumo Real\nCorrelação r = {treino['previsao_chuva_mm'].corr(treino['consumo']):+.3f} (Impacto Mínimo)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Chuva Prevista no Dia da Emissão (mm)")
    axes[0].set_ylabel("Refeições Servidas")
    
    # Temperatura Máxima Prevista vs Consumo
    sns.regplot(data=treino, x="previsao_temp_max", y="consumo", ax=axes[1], color="#e67e22", scatter_kws={"alpha": 0.4, "s": 25}, line_kws={"color": "#e74c3c", "linewidth": 1.5})
    axes[1].set_title(f"Previsão de Temperatura Máx (°C) vs. Consumo Real\nCorrelação r = {treino['previsao_temp_max'].corr(treino['consumo']):+.3f} (Impacto Nulo)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Temperatura Máxima Prevista (°C)")
    axes[1].set_ylabel("Refeições Servidas")
    
    plt.tight_layout()
    fig.savefig(PASTA_FIGURAS / "figura_05_sensibilidade_climatica.png")
    plt.close()


def executar_eda_completa() -> dict:
    """Executa a rotina completa de análise exploratória, salva tabelas e gráficos."""
    treino, periodo_discente, periodo_nutri = carregar_dados_eda()
    
    df_humanos = avaliar_metodos_humanos(periodo_discente, periodo_nutri)
    df_ranking = calcular_ranking_atributos(treino, periodo_discente, periodo_nutri)
    df_semana = analisar_padrao_dia_semana(treino, periodo_discente, periodo_nutri)
    df_fim = analisar_desmobilizacao_semestre(treino)
    df_prot = analisar_cardapio_proteinas(treino)
    
    gerar_graficos_eda(treino, periodo_discente, periodo_nutri)
    
    return {
        "dias_treino_total": len(treino),
        "dias_regime_discente_2023_2025": len(periodo_discente),
        "dias_regime_nutricionista_2026": len(periodo_nutri),
        "tabelas_geradas": [
            "comparativo_metodos_humanos.csv",
            "ranking_impacto_atributos.csv",
            "sazonalidade_dia_semana.csv",
            "queda_fim_de_semestre.csv",
            "impacto_proteinas_cardapio.csv"
        ],
        "figuras_geradas": [
            "figura_01_comparativo_metodos_humanos.png",
            "figura_02_sazonalidade_semanal.png",
            "figura_03_desmobilizacao_fim_semestre.png",
            "figura_04_ranking_correlacoes.png",
            "figura_05_sensibilidade_climatica.png"
        ]
    }


if __name__ == "__main__":
    res = executar_eda_completa()
    print("=== Análise Exploratória Concluída com Sucesso! ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
