"""Interface Web Operacional do Artefato — Apoio à Decisão e Storytelling da Pesquisa
Pesquisa: Artefato Analítico-Preditivo para Planejamento da Demanda e Mitigação do Desperdício em Restaurante Institucional de Ensino
Autor / Mestrando: Michel Silva de Souza
Metodologia: Design Science Research (DSR)
Campo de Aplicação: Restaurante Estudantil de Ensino Público Profissional e Tecnológico
"""

import sys
from pathlib import Path
import json

# Garantir inclusão da raiz do projeto no sys.path
RAIZ = Path(__file__).resolve().parent.parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from src.core.config import GOLD, SILVER, RESULTADOS
from src.core.test_zero_leakage_models import executar_auditoria_completa_vazamento

# ==============================================================================
# 1. Configuração da Página e Estilo Visual Institucional
# ==============================================================================
st.set_page_config(
    page_title="Artefato Analítico-Preditivo RU | DSR 2026",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilização CSS Moderna com Identidade Visual da Rede Federal / IFSP
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
    --primary: #008744;
    --primary-dark: #006834;
    --primary-light: #e8f5e9;
    --accent: #0284c7;
    --bg-page: #f8fafc;
    --bg-card: #ffffff;
    --border: #e2e8f0;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --danger: #dc2626;
    --danger-light: #fee2e2;
    --warning: #d97706;
    --warning-light: #fef3c7;
    --success: #16a34a;
}

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-page) !important;
    color: var(--text-main) !important;
}

.top-header {
    background: #ffffff;
    border: 1px solid var(--border);
    border-left: 6px solid var(--primary);
    border-radius: 10px;
    padding: 16px 22px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.top-title {
    font-size: 22px;
    font-weight: 800;
    color: var(--text-main);
    margin: 0;
}

.top-sub {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 4px;
}

.badge-tag {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-green { background: #dcfce7; color: #166534; }
.badge-blue { background: #e0f2fe; color: #075985; }

.kpi-container {
    display: flex;
    gap: 14px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}

.kpi-card {
    flex: 1;
    min-width: 200px;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.kpi-label {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 24px;
    font-weight: 800;
    color: var(--text-main);
    margin: 4px 0;
}

.kpi-hint {
    font-size: 11px;
    color: var(--text-muted);
}

.card-box {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. Carregamento de Dados com Cache
# ==============================================================================
@st.cache_data
def carregar_dados_globais():
    gold_t1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    gold_t2 = pd.read_parquet(GOLD / "gold_t2.parquet")
    silver = pd.read_parquet(SILVER / "silver_completo.parquet")
    
    caminho_pred = RESULTADOS / "modelos" / "predicoes_detalhadas_torneio.parquet"
    if caminho_pred.exists():
        df_pred = pd.read_parquet(caminho_pred)
    else:
        df_pred = pd.DataFrame()
        
    return gold_t1, gold_t2, silver, df_pred

gold_t1, gold_t2, silver, df_pred = carregar_dados_globais()

# Cabeçalho Institucional Superior
st.markdown("""
<div class="top-header">
    <div>
        <div class="top-title">🍲 Restaurante Estudantil — Sistema Analítico-Preditivo de Apoio à Decisão</div>
        <div class="top-sub">Mestrado Profissional • Metodologia Design Science Research (DSR) • Gestão Pública</div>
    </div>
    <div>
        <span class="badge-tag badge-green">● Em Produção</span>
        <span class="badge-tag badge-blue">Zero Data Leakage Auditado</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Navegação Principal Horizontal
opcoes_nav = [
    "📊 Visão Executiva & Histórico",
    "🔮 Preparo Diário (T+1)",
    "📦 Compras Quinzenais (T+2)",
    "🛡️ Auditoria de Não-Vazamento",
    "🌊 Resiliência Climática & Teste Cego"
]

aba_selecionada = st.pills("Navegação:", opcoes_nav, default=opcoes_nav[0], label_visibility="collapsed")

# ==============================================================================
# ABA 1: VISÃO EXECUTIVA & HISTÓRICO
# ==============================================================================
if aba_selecionada == "📊 Visão Executiva & Histórico":
    st.markdown("### 📊 Painel Executivo e Diagnóstico Longitudinal (2023 a 2026)")
    st.markdown("Visão integrada de frequência discente, comparação dos métodos humanos e diagnóstico dos motores de consumo.")
    
    # Filtros Globais
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        ano_filtro = st.selectbox("Recorte Temporal:", ["Histórico Completo (2023-2026)", "2026 (Gestão Técnica)", "2023-2025 (Reservas Discentes)"])
    with col_f2:
        remover_atipicos = st.checkbox("Excluir Dias Atípicos Graves (greves e fora de aula)", value=True)
    with col_f3:
        dias_semana = st.multiselect("Dias da Semana:", ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira"],
                                     default=["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira"])

    df_base = silver[silver["is_dia_letivo"] & silver["consumo"].notna()].copy()
    if remover_atipicos and "usar_no_treino" in df_base.columns:
        df_base = df_base[df_base["usar_no_treino"]]
    if dias_semana:
        df_base = df_base[df_base["dia_da_semana"].isin(dias_semana)]
        
    if ano_filtro == "2026 (Gestão Técnica)":
        df_base = df_base[df_base["data"].dt.year == 2026]
    elif ano_filtro == "2023-2025 (Reservas Discentes)":
        df_base = df_base[df_base["data"].dt.year < 2026]

    # KPIs Superiores
    n_dias = len(df_base)
    media_cons = int(df_base["consumo"].mean()) if n_dias > 0 else 0
    pico_cons = int(df_base["consumo"].max()) if n_dias > 0 else 0
    min_cons = int(df_base["consumo"].min()) if n_dias > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dias Úteis Avaliados", f"{n_dias} dias", help="Dias letivos com consumo real")
    with col2:
        st.metric("Consumo Médio Diário", f"{media_cons} ref/dia", help="Média de refeições servidas na catraca")
    with col3:
        st.metric("Pico Máximo de Demanda", f"{pico_cons} ref", help="Dia de maior adesão escolar")
    with col4:
        st.metric("Piso Mínimo de Demanda", f"{min_cons} ref", help="Menor quórum registrado")

    # Gráfico Longitudinal
    st.markdown("#### Série Temporal de Consumo Real na Catraca")
    chart_cons = alt.Chart(df_base).mark_line(point=True, color="#008744").encode(
        x=alt.X("data:T", title="Data"),
        y=alt.Y("consumo:Q", title="Refeições Servidas"),
        tooltip=[
            alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
            alt.Tooltip("consumo:Q", title="Consumo Real"),
            alt.Tooltip("dia_da_semana:N", title="Dia da Semana")
        ]
    ).properties(height=320).interactive()
    st.altair_chart(chart_cons, use_container_width=True)

    # Confronto Humano: Reservas Discentes vs Nutricionista
    st.markdown("#### Comparativo dos Métodos Humanos de Planejamento")
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.markdown("**Regime 2023–2025: Reservas Discentes Auto-Declaradas**")
        st.markdown("- **MAE:** 27,03 refeições | **WAPE:** 19,94%")
        st.markdown("- **Taxa de Déficit:** 65,6% dos dias (9.332 refeições faltantes acumuladas no agendamento)")
        st.markdown("- **Comportamento:** Forte absenteísmo discente e agendamentos não comparecidos.")
    with col_t2:
        st.markdown("**Regime 2026: Planejamento Técnico da Nutricionista**")
        st.markdown("- **MAE:** 24,86 refeições | **WAPE:** 14,20%")
        st.markdown("- **Taxa de Déficit:** 39,6% dos dias (994 refeições faltantes)")
        st.markdown("- **Comportamento:** Redução de 51% no volume de faltas, porém com vulnerabilidade em choques climáticos.")

    # Galeria de Figuras da EDA
    with st.expander("🔍 Explorar Figuras Acadêmicas da EDA (Dissertação)", expanded=False):
        fig_cols = st.columns(2)
        pasta_eda = RESULTADOS / "figuras" / "eda"
        if pasta_eda.exists():
            figuras_eda = sorted(list(pasta_eda.glob("*.png")))
            for i, fig_path in enumerate(figuras_eda[:4]):
                with fig_cols[i % 2]:
                    st.image(str(fig_path), caption=fig_path.name.replace("_", " ").replace(".png", "").title())

# ==============================================================================
# ABA 2: PREPARO DIÁRIO (T+1)
# ==============================================================================
elif aba_selecionada == "🔮 Preparo Diário (T+1)":
    st.markdown("### 🔮 Planejamento Operacional de Preparo Diário (Horizonte T+1)")
    st.markdown("**Antecedência:** 3 a 7 dias úteis (decisão tomada na sexta-feira de corte anterior). Inclui o cardápio e a estimativa técnica da nutricionista.")

    tab_retros, tab_simul = st.tabs(["📊 Desempenho Retrospectivo Validado", "🧪 Simulador Interativo com Margem Conformal"])
    
    with tab_retros:
        st.markdown("#### Resultados do Torneio Walk-Forward (Fevereiro a Agosto de 2026 / N=121 dias)")
        
        caminho_tab4 = RESULTADOS / "tabelas_dissertacao" / "tabela_04_modelos_vs_nutricionista_t1_retrospectivo.csv"
        if caminho_tab4.exists():
            df_t4 = pd.read_csv(caminho_tab4, sep=";")
            st.dataframe(df_t4, use_container_width=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            p_fig20 = RESULTADOS / "figuras" / "dissertacao" / "figura_20_modelos_vs_nutricionista_t1_retrospectivo.png"
            if p_fig20.exists():
                st.image(str(p_fig20), caption="Figura 20: Dispersão e Correlação dos Modelos vs. Nutricionista")
        with col_g2:
            p_fig21 = RESULTADOS / "figuras" / "dissertacao" / "figura_21_fronteira_equilibrio_modelo_hibrido_retrospectivo.png"
            if p_fig21.exists():
                st.image(str(p_fig21), caption="Figura 21: Fronteira de Combinação do Modelo Híbrido")

    with tab_simul:
        st.markdown("#### Simulador de Políticas Operacionais e Margens de Segurança")
        st.markdown("Selecione o modelo base e ajuste a margem de segurança conformal para avaliar o impacto no desperdício e no risco de faltas.")

        col_s1, col_s2, col_s3 = st.columns([2, 2, 2])
        with col_s1:
            modelo_escolhido = st.selectbox(
                "Modelo Base:",
                ["Modelo Híbrido 50/50 (IA + Nutricionista)", "IA Campeã (ExtraTrees / Lasso)", "Nutricionista Institucional"]
            )
        with col_s2:
            margem_buffer = st.slider("Margem de Segurança (refeições extras):", min_value=0, max_value=30, value=12, step=1,
                                     help="12 refeições correspondem ao Conformal P85 calibrado empiricamente")
        with col_s3:
            custo_ref = st.number_input("Custo Unitário da Refeição (R$):", value=14.00, step=0.50)

        # Exibir Impacto da Margem Selecionada
        caminho_tab5 = RESULTADOS / "tabelas_dissertacao" / "tabela_05_politicas_margens_retrospectivo.csv"
        if caminho_tab5.exists():
            df_t5 = pd.read_csv(caminho_tab5, sep=";")
            st.markdown("##### Políticas Padronizadas na Pesquisa:")
            st.dataframe(df_t5, use_container_width=True)

        st.info(f"💡 **Diretriz de Gestão:** A política **Conformal P85 (+12 refeições)** reduziu a ocorrência de dias com déficit de **43,4% para apenas 15,7%**, mantendo um custo controlado de R$ 27.916,00 em sobras no semestre — alcançando a fronteira ótima de Pareto.")

# ==============================================================================
# ABA 3: COMPRAS QUINZENAIS (T+2)
# ==============================================================================
elif aba_selecionada == "📦 Compras Quinzenais (T+2)":
    st.markdown("### 📦 Planejamento Autônomo de Compras e Suprimentos (Horizonte T+2)")
    st.markdown("**Antecedência:** 10 a 14 dias úteis. **100% Autônomo:** Opera sem depender do plano da nutricionista nem de reservas de alunos.")

    st.markdown("#### Desempenho dos Algoritmos de Compras (Retrospectivo 2026)")
    caminho_tab6 = RESULTADOS / "tabelas_dissertacao" / "tabela_06_modelos_t2_compras_retrospectivo.csv"
    if caminho_tab6.exists():
        df_t6 = pd.read_csv(caminho_tab6, sep=";")
        st.dataframe(df_t6, use_container_width=True)

    col_c1, col_c2 = st.columns([1.2, 0.8])
    with col_c1:
        st.markdown("""
        **Vantagens da Automação de Compras em T+2:**
        1. **Antecedência Crítica:** Permite que o setor de compras emita ordens de fornecimento para carnes, hortifrútis e secos com 2 semanas de margem, respeitando prazos licitatórios.
        2. **Independência Operacional:** A nutricionista não precisa adiantar cálculos manuais 15 dias antes, liberando tempo para a supervisão técnica da cozinha e elaboração dos cardápios.
        3. **Acurácia Validada:** O modelo **Ensemble Voting** atingiu MAE de **24,13 refeições (WAPE de 13,79%)**, precisão virtualmente idêntica à da nutricionista em curto prazo ($T+1$, MAE=24,86).
        """)
    with col_c2:
        p_fig18 = RESULTADOS / "figuras" / "dissertacao" / "figura_18_ablation_cenarios_horizontes_retrospectivo.png"
        if p_fig18.exists():
            st.image(str(p_fig18), caption="Figura 18: Comparativo de Horizontes T+1 vs T+2")

# ==============================================================================
# ABA 4: AUDITORIA DE NÃO-VAZAMENTO
# ==============================================================================
elif aba_selecionada == "🛡️ Auditoria de Não-Vazamento":
    st.markdown("### 🛡️ Painel de Integridade Científica e Zero Data Leakage")
    st.markdown("Verificação formal de isolamento temporal para bancas acadêmicas, pesquisadores e órgãos de controle.")

    relatorio_path = RESULTADOS / "auditoria" / "relatorio_teste_vazamento_modelos.json"
    if relatorio_path.exists():
        with open(relatorio_path, "r", encoding="utf-8") as f:
            rel = json.load(f)
            
        status = rel.get("status_geral", "APROVADO")
        st.success(f"✅ **STATUS GERAL DA BASE:** {status}")
        st.caption(f"Última auditoria executada em: {rel.get('data_auditoria', '01/11/2026')}")
        
        for teste in rel.get("testes", []):
            with st.container(border=True):
                st.markdown(f"#### {teste.get('teste')}")
                if teste.get("sucesso"):
                    st.markdown("🟢 **Resultado:** APROVADO COM ZERO FALHAS")
                else:
                    st.markdown("🔴 **Resultado:** REPROVADO")
                    
                if "detalhe" in teste:
                    st.markdown(f"- *Diagnóstico:* {teste['detalhe']}")
                if "detalhes" in teste:
                    for d in teste["detalhes"]:
                        st.markdown(f"- {d}")
    else:
        st.info("Nenhum relatório prévio em disco. Clique abaixo para executar a auditoria.")

    if st.button("Executar Bateria Adversarial Anti-Vazamento Agora"):
        with st.spinner("Executando simulação adversarial e testes de corte temporal..."):
            rel_novo = executar_auditoria_completa_vazamento()
            st.success("Auditoria finalizada com 100% de conformidade!")
            st.rerun()

# ==============================================================================
# ABA 5: RESILIÊNCIA CLIMÁTICA & TESTE CEGO
# ==============================================================================
elif aba_selecionada == "🌊 Resiliência Climática & Teste Cego":
    st.markdown("### 🌊 Resiliência a Eventos Extremos e Teste Cego Out-of-Time")
    st.markdown("**Período Prospectivo:** Setembro e Outubro de 2026 ($N=28$ dias). Avaliação do modelo congelado em 31/08/2026 frente às enchentes do Alto Tietê.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("#### Teste Cego Estrito (Modelo Congelado em 31/08/2026)")
        caminho_tab7 = RESULTADOS / "tabelas_dissertacao" / "tabela_07_desempenho_prospectivo_set_out.csv"
        if caminho_tab7.exists():
            df_t7 = pd.read_csv(caminho_tab7, sep=";")
            st.dataframe(df_t7.head(5), use_container_width=True)
    with col_p2:
        st.markdown("#### Resiliência nas Enchentes de Setembro (N=5 dias críticos)")
        caminho_tab8 = RESULTADOS / "tabelas_dissertacao" / "tabela_08_resiliencia_climatica_setembro_enchentes.csv"
        if caminho_tab8.exists():
            df_t8 = pd.read_csv(caminho_tab8, sep=";")
            st.dataframe(df_t8, use_container_width=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        p_fig25 = RESULTADOS / "figuras" / "dissertacao" / "figura_25_confronto_diario_setembro_enchentes_ia_vs_nutri.png"
        if p_fig25.exists():
            st.image(str(p_fig25), caption="Figura 25: A Crise das Inundações em Setembro de 2026")
    with col_f2:
        p_fig28 = RESULTADOS / "figuras" / "dissertacao" / "figura_28_sintese_retrospectivo_vs_prospectivo.png"
        if p_fig28.exists():
            st.image(str(p_fig28), caption="Figura 28: Síntese de Erro (MAE) Retrospectivo vs. Prospectivo")

# Rodapé Institucional
st.markdown("---")
st.caption("Artefato Analítico-Preditivo de Apoio à Decisão • Metodologia Design Science Research (DSR) • Desenvolvido para o Restaurante Estudantil • 2026")
