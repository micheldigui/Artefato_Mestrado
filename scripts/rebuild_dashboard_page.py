"""Script para reconstruir a página 1 (Dashboard) com todos os gráficos interativos da EDA,
seções temáticas separadas e total responsividade para mobile, tablet e desktop.
"""
from pathlib import Path

app_path = Path("src/app/main.py")
content = app_path.read_text(encoding="utf-8")

# Identificar limites da Página 1: Dashboard
p1_start = content.find('if pagina_ativa == "Dashboard":')
p2_start = content.find('elif pagina_ativa == "Previs')

if p1_start == -1 or p2_start == -1:
    raise ValueError("Marcadores não encontrados no arquivo main.py")

parte_antes = content[:p1_start]
parte_depois = content[p2_start:]

novo_dashboard = '''if pagina_ativa == "Dashboard":
    st.markdown('<div class="story-badge">ETAPA 1 DE 6 • DIAGNÓSTICO HISTÓRICO & EDA COMPLETA</div>', unsafe_allow_html=True)
    st.markdown("### Visão Geral, Indicadores e Análise Exploratória (EDA)")
    st.markdown('<p class="story-subtitle">Explore de forma 100% interativa todas as evidências que regem a frequência discente: consumo histórico, calendário letivo, cardápio, variáveis climáticas e o ranking de poder preditivo.</p>', unsafe_allow_html=True)
    
    renderizar_painel_explicativo(
        objetivo_rotina=(
            "**Visualizar o comportamento real dos alunos no refeitório e embasar decisões com dados.** "
            "Aqui você descobre exatamente o que faz a quantidade de comida necessária subir ou descer, "
            "comparando o que foi servido contra as reservas dos alunos e as previsões da equipe."
        ),
        como_foi_feito_ciencia=(
            "**Estudo de mais de 650 dias letivos reais auditados na instituição (2023 a 2026).** "
            "A pesquisa comprovou que a presença discente não é mero acaso: ela responde à proteína servida, "
            "ao dia da semana, à proximidade do fim do semestre e às tempestades severas."
        ),
        como_usar_passos=(
            "1. **Use os filtros interativos** para recortar anos ou dias da semana desejados.\\n"
            "2. **Selecione uma das 5 seções temáticas** abaixo para navegar pelos gráficos interativos.\\n"
            "3. **Passe o mouse ou toque nos gráficos** para inspecionar os números exatos e detalhes de cada ponto."
        )
    )
    
    # Carregamento da base completa
    df_fato_raw = carregar_fato_diaria_completa()
    df_validos_geral = df_fato_raw[
        (df_fato_raw["is_dia_letivo"] == True) & 
        (df_fato_raw["consumo"].notna()) & 
        (df_fato_raw["consumo"] > 0)
    ].copy()
    df_validos_geral = df_validos_geral.sort_values("data_dt")
    
    # Linha de Filtros Interativos Globais do Dashboard
    with st.container(border=True):
        st.markdown("##### 🎛️ Filtros Interativos do Histórico")
        col_f1, col_f2 = st.columns([1.5, 2.5])
        with col_f1:
            filtro_ano = st.selectbox(
                "Filtrar por Período / Ano Letivo:",
                ["Todo o Histórico (2023–2026)", "2026 (Ano Corrente)", "2025", "2024", "2023"],
                index=0
            )
        with col_f2:
            opcoes_dias = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira"]
            filtro_dias_sem = st.multiselect(
                "Filtrar por Dias da Semana:",
                opcoes_dias,
                default=opcoes_dias,
                format_func=lambda x: x.capitalize()
            )
            
        ocultar_atipicos = st.checkbox(
            "Desconsiderar dias atípicos / fora da curva (< 30 refeições em encerramentos e testes)",
            value=True,
            help="Desconsidera registros pontuais com consumo inferior a 30 refeições (como semanas de exames finais, conselhos e testes de sistema), revelando a menor presença em dias de operação escolar regular."
        )
            
    # Aplicação dos filtros dinâmicos com normalização robusta
    df_dash = df_validos_geral.copy()
    if filtro_ano == "2026 (Ano Corrente)":
        df_dash = df_dash[df_dash["ano"] == 2026]
    elif filtro_ano == "2025":
        df_dash = df_dash[df_dash["ano"] == 2025]
    elif filtro_ano == "2024":
        df_dash = df_dash[df_dash["ano"] == 2024]
    elif filtro_ano == "2023":
        df_dash = df_dash[df_dash["ano"] == 2023]
        
    def normalizar_dia_chave(d):
        txt = str(d).lower().strip()
        for k in ["seg", "ter", "qua", "qui", "sex", "sab", "dom"]:
            if k in txt:
                return k
        return txt

    if filtro_dias_sem:
        chaves_sel = [normalizar_dia_chave(d) for d in filtro_dias_sem]
        df_dash["_dia_k"] = df_dash["dia_da_semana"].apply(normalizar_dia_chave)
        df_dash = df_dash[df_dash["_dia_k"].isin(chaves_sel)]
    else:
        df_dash = df_dash.iloc[0:0]

    if ocultar_atipicos:
        df_dash = df_dash[df_dash["consumo"] >= 30]
        
    if df_dash.empty:
        st.warning("⚠️ **Nenhum registro encontrado para este filtro.** Selecione ao menos um dia da semana para visualizar os gráficos e indicadores.")
        st.stop()

    # KPIs Dinâmicos com base nos filtros
    tot_dias = len(df_dash)
    med_cons = int(round(df_dash["consumo"].mean())) if tot_dias > 0 else 0
    max_cons = int(df_dash["consumo"].max()) if tot_dias > 0 else 0
    min_cons = int(df_dash["consumo"].min()) if tot_dias > 0 else 0
    
    sub_min = "Menor quórum em aula regular" if ocultar_atipicos else "Dia de menor adesão discente"
    sub_tot = "Dias regulares (sem atípicos)" if ocultar_atipicos else "Dias letivos válidos no filtro"
    
    kpi_html = (
        f'<div class="kpi-row">'
        f'<div class="kpi-card b-primary"><div class="kpi-title">Média de Refeições Servidas</div><div class="kpi-val">{med_cons} ref</div><div class="kpi-sub">Frequência média no recorte</div></div>'
        f'<div class="kpi-card b-success"><div class="kpi-title">Pico Máximo de Presença</div><div class="kpi-val">{max_cons} ref</div><div class="kpi-sub">Maior consumo registrado</div></div>'
        f'<div class="kpi-card b-warning"><div class="kpi-title">Menor Presença (Piso)</div><div class="kpi-val">{min_cons} ref</div><div class="kpi-sub">{sub_min}</div></div>'
        f'<div class="kpi-card b-accent"><div class="kpi-title">Total de Datas Analisadas</div><div class="kpi-val">{tot_dias} dias</div><div class="kpi-sub">{sub_tot}</div></div>'
        f'</div>'
    )
    st.markdown(kpi_html, unsafe_allow_html=True)
    
    # Seletor Moderno das 5 Seções Analíticas do Dashboard
    st.markdown("##### 🧭 Escolha a Seção Temática que Deseja Explorar:")
    opcoes_secoes = [
        "📊 1. Consumo Histórico & Regimes",
        "📅 2. Calendário & Sazonalidade",
        "🍲 3. Cardápio & Atratividade",
        "🌦️ 4. Clima & Efeito Meteorológico",
        "🌪️ 5. Poder Preditivo dos Atributos (EDA)"
    ]
    if hasattr(st, "pills"):
        secao_dash = st.pills("Seção Temática:", opcoes_secoes, default=opcoes_secoes[0], label_visibility="collapsed")
    elif hasattr(st, "segmented_control"):
        secao_dash = st.segmented_control("Seção Temática:", opcoes_secoes, default=opcoes_secoes[0], label_visibility="collapsed")
    else:
        secao_dash = st.radio("Seção Temática:", opcoes_secoes, horizontal=True, label_visibility="collapsed")
    
    if not secao_dash:
        secao_dash = opcoes_secoes[0]
        
    st.write("")
    
    # --------------------------------------------------------------------------
    # SUBSEÇÃO 1: REGIMES DE GESTÃO VS CONSUMO REAL
    # --------------------------------------------------------------------------
    if "1. Consumo Histórico" in secao_dash:
        with st.container(border=True):
            st.markdown("#### ⚖️ Regimes Históricos de Gestão vs. Consumo Real dos Alunos")
            st.write(
                "**Entenda a história do nosso refeitório:** "
                "Entre 2023 e 2025, a quantidade de comida dependia do preenchimento voluntário de reservas feitas pelos discentes no sistema acadêmico. "
                "Muitos esqueciam de registrar: em **65,6% dos dias faltavam refeições previstas no agendamento**, gerando risco de estudantes ficarem sem almoço. "
                "Já em 2026, a nutrição passou a estipular um quantitativo técnico com folga preventiva, "
                "o que reduziu a taxa de falta para 39,6%, mas gerou sobras de segurança em dias de baixa presença. "
                "O artefato preditivo une os dois mundos para equilibrar perfeitamente a balança."
            )
            
            df_reg = df_dash.copy()
            col_nutri = df_reg["planejado_nutri"] if "planejado_nutri" in df_reg.columns else pd.Series(index=df_reg.index, dtype=float)
            col_res = df_reg["reservas"] if "reservas" in df_reg.columns else pd.Series(index=df_reg.index, dtype=float)
            df_reg["referencia_planejada"] = col_nutri.combine_first(col_res)
            df_reg = df_reg.dropna(subset=["referencia_planejada"]).copy()
            df_reg["desvio"] = df_reg["referencia_planejada"] - df_reg["consumo"]
            df_reg["desvio_acum"] = df_reg["desvio"].cumsum()
            df_reg["tipo_desvio"] = df_reg["desvio"].apply(
                lambda x: "Sobra / Excedente (+)" if x > 0 else ("Falta Potencial (-)" if x < 0 else "Equilíbrio Exato")
            )
            
            # Linha 1 de Gráficos: Balanço Diário e Linha do Tempo Temporal
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown("##### Desvios Diários (Onde Sobrou e Onde Faltou Comida)")
                st.caption("Barras acima de zero indicam sobras; barras abaixo indicam déficit em relação à presença real.")
                
                slice_tail = df_reg.tail(120) if len(df_reg) > 120 else df_reg
                chart_desv = alt.Chart(slice_tail).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X("data_dt:T", title=None, axis=alt.Axis(format="%d/%m", labelAngle=-45)),
                    y=alt.Y("desvio:Q", title="Diferença (Planejado - Consumo Real)"),
                    color=alt.Color(
                        "tipo_desvio:N",
                        scale=alt.Scale(
                            domain=["Sobra / Excedente (+)", "Falta Potencial (-)"],
                            range=["#f59e0b", "#ef4444"]
                        ),
                        legend=alt.Legend(title="Balanço Diário", orient="top")
                    ),
                    tooltip=[
                        alt.Tooltip("data:N", title="Data"),
                        alt.Tooltip("consumo:Q", title="Consumo Real (alunos)"),
                        alt.Tooltip("referencia_planejada:Q", title="Previsto / Reservado"),
                        alt.Tooltip("desvio:Q", title="Diferença (ref)")
                    ]
                ).properties(height=280).interactive()
                st.altair_chart(chart_desv, use_container_width=True)
                
            with col_r2:
                st.markdown("##### Linha do Tempo: Demanda Real vs. Quantitativo Planejado")
                st.caption("Consumo real (linha verde) comparado ao quantitativo previsto/reservas (linha tracejada âmbar).")
                
                chart_real = alt.Chart(slice_tail).mark_line(color="#10b981", strokeWidth=2.2).encode(
                    x=alt.X("data_dt:T", title=None, axis=alt.Axis(format="%d/%m", labelAngle=-45)),
                    y=alt.Y("consumo:Q", title="Total de Refeições"),
                    tooltip=[
                        alt.Tooltip("data:N", title="Data"),
                        alt.Tooltip("consumo:Q", title="Consumo Real (alunos)")
                    ]
                )
                chart_plan = alt.Chart(slice_tail).mark_line(color="#f59e0b", strokeDash=[4, 4], strokeWidth=2).encode(
                    x=alt.X("data_dt:T", title=None),
                    y=alt.Y("referencia_planejada:Q"),
                    tooltip=[
                        alt.Tooltip("data:N", title="Data"),
                        alt.Tooltip("referencia_planejada:Q", title="Planejado / Reserva")
                    ]
                )
                chart_pts = alt.Chart(slice_tail).mark_circle(size=25, color="#10b981").encode(
                    x=alt.X("data_dt:T"),
                    y=alt.Y("consumo:Q"),
                    tooltip=[
                        alt.Tooltip("data:N", title="Data"),
                        alt.Tooltip("consumo:Q", title="Consumo Real"),
                        alt.Tooltip("referencia_planejada:Q", title="Planejado / Reserva"),
                        alt.Tooltip("desvio:Q", title="Saldo (ref)")
                    ]
                )
                chart_timeline = (chart_real + chart_plan + chart_pts).properties(height=280).interactive()
                st.altair_chart(chart_timeline, use_container_width=True)

            # Linha 2 de Análises: Balanço Anual de Falta vs Sobra e Dispersão Interativa 45°
            col_r3, col_r4 = st.columns(2)
            with col_r3:
                st.markdown("##### Potencial de Falta (Déficit) vs. Sobra por Ano")
                st.caption("Comparativo anual consolidado de refeições faltantes vs. excedentes acumuladas (Dissertação Fig. 02).")
                
                df_bal_base = df_validos_geral.copy()
                c_nut = df_bal_base["planejado_nutri"] if "planejado_nutri" in df_bal_base.columns else pd.Series(index=df_bal_base.index, dtype=float)
                c_res = df_bal_base["reservas"] if "reservas" in df_bal_base.columns else pd.Series(index=df_bal_base.index, dtype=float)
                df_bal_base["referencia_bal"] = c_nut.combine_first(c_res)
                df_bal_base = df_bal_base.dropna(subset=["referencia_bal"]).copy()
                
                df_bal_base["falta_ref"] = (df_bal_base["consumo"] - df_bal_base["referencia_bal"]).clip(lower=0)
                df_bal_base["sobra_ref"] = (df_bal_base["referencia_bal"] - df_bal_base["consumo"]).clip(lower=0)
                
                df_bal_ano = df_bal_base.groupby("ano")[["sobra_ref", "falta_ref"]].sum().reset_index()
                df_bal_melt = df_bal_ano.melt(id_vars="ano", value_vars=["sobra_ref", "falta_ref"], var_name="tipo", value_name="volume")
                df_bal_melt["tipo_label"] = df_bal_melt["tipo"].map({"sobra_ref": "Sobras Potenciais (Excedente)", "falta_ref": "Déficit Potencial (Falta no Agendamento)"})
                df_bal_melt["ano_str"] = df_bal_melt["ano"].astype(str)
                
                chart_bal = alt.Chart(df_bal_melt).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("ano_str:N", title="Ano Letivo", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("volume:Q", title="Volume de Refeições Acumuladas"),
                    color=alt.Color("tipo_label:N", scale=alt.Scale(domain=["Sobras Potenciais (Excedente)", "Déficit Potencial (Falta no Agendamento)"], range=["#f59e0b", "#ef4444"]), legend=alt.Legend(title=None, orient="top")),
                    xOffset="tipo_label:N",
                    tooltip=[
                        alt.Tooltip("ano_str:N", title="Ano"),
                        alt.Tooltip("tipo_label:N", title="Tipo de Saldo"),
                        alt.Tooltip("volume:Q", title="Total de Refeições", format=",.0f")
                    ]
                ).properties(height=280).interactive()
                st.altair_chart(chart_bal, use_container_width=True)
                
            with col_r4:
                st.markdown("##### Dispersão: Consumo Real vs. Referência Planejada")
                st.caption("Pontos próximos da linha diagonal de 45° indicam alta precisão; dispersão indica erro.")
                
                df_disp = df_dash.copy()
                c_n = df_disp["planejado_nutri"] if "planejado_nutri" in df_disp.columns else pd.Series(index=df_disp.index, dtype=float)
                c_r = df_disp["reservas"] if "reservas" in df_disp.columns else pd.Series(index=df_disp.index, dtype=float)
                df_disp["planejado"] = c_n.combine_first(c_r)
                df_disp = df_disp.dropna(subset=["planejado", "consumo"]).copy()
                df_disp["regime_nome"] = df_disp["ano"].apply(lambda a: "Nutricionista (2026)" if a == 2026 else "Reservas Alunos (2023-2025)")
                
                scatter_pts = alt.Chart(df_disp).mark_circle(size=45, opacity=0.6).encode(
                    x=alt.X("planejado:Q", title="Quantitativo Planejado / Reservado (ref)"),
                    y=alt.Y("consumo:Q", title="Consumo Real na Catraca (ref)"),
                    color=alt.Color("regime_nome:N", scale=alt.Scale(domain=["Nutricionista (2026)", "Reservas Alunos (2023-2025)"], range=["#008744", "#0284c7"]), legend=alt.Legend(title="Regime", orient="top")),
                    tooltip=[
                        alt.Tooltip("data:N", title="Data"),
                        alt.Tooltip("planejado:Q", title="Planejado"),
                        alt.Tooltip("consumo:Q", title="Real"),
                        alt.Tooltip("regime_nome:N", title="Regime")
                    ]
                )
                
                # Linha de 45 graus (identidade)
                max_v = max(df_disp["planejado"].max(), df_disp["consumo"].max()) if len(df_disp) > 0 else 300
                df_diag = pd.DataFrame({"x": [0, max_v], "y": [0, max_v]})
                diag_line = alt.Chart(df_diag).mark_line(color="#94a3b8", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
                
                st.altair_chart((scatter_pts + diag_line).properties(height=280).interactive(), use_container_width=True)

            # Expander com Figuras Acadêmicas Oficiais da Dissertação
            with st.expander("🖼️ Ver Figuras Acadêmicas da Dissertação (Figuras 01 e 02)", expanded=False):
                col_f_a, col_f_b = st.columns(2)
                with col_f_a:
                    p1 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_01_confronto_metodos_humanos.png")
                    if os.path.exists(p1):
                        st.image(p1, caption="Figura 01: Confronto dos Métodos Humanos (Reservas Discentes vs. Nutricionista)", use_container_width=True)
                with col_f_b:
                    p2 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_02_balanco_sobras_faltas.png")
                    if os.path.exists(p2):
                        st.image(p2, caption="Figura 02: Balanço Acumulado de Sobras vs Faltas Potenciais", use_container_width=True)

    # --------------------------------------------------------------------------
    # SUBSEÇÃO 2: CALENDÁRIO & SAZONALIDADE ESCOLAR
    # --------------------------------------------------------------------------
    elif "2. Calendário" in secao_dash:
        with st.container(border=True):
            st.markdown("#### 📅 O Ritmo Semanal, Mensal e Desmobilização de Fim de Semestre")
            st.write(
                "**A dinâmica temporal das aulas:** "
                "A presença discente não é uniforme: a segunda-feira começa com alta frequência, "
                "a quarta-feira atinge o pico com disciplinas técnicas de laboratório integradas e a sexta-feira à tarde sofre uma evasão natural de cerca de 20%. "
                "Além disso, a pesquisa descobriu que **nos últimos 3 dias úteis do semestre, o consumo desaba 56,1%** devido ao encerramento antecipado de avaliações."
            )
            
            dias_ordem = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira"]
            dias_map = {"segunda-feira": "Segunda", "terça-feira": "Terça", "quarta-feira": "Quarta", "quinta-feira": "Quinta", "sexta-feira": "Sexta"}
            meses_map = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun", 7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}
            
            df_d = df_dash[df_dash["dia_da_semana"].isin(dias_ordem)].groupby("dia_da_semana")["consumo"].agg(["count", "mean"]).reset_index()
            df_d["dia_nome"] = df_d["dia_da_semana"].map(dias_map)
            df_d["consumo_medio"] = df_d["mean"].round(1)
            
            # Linha 1: Média por Dia da Semana e Sazonalidade Mensal
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("##### Média de Presença por Dia da Semana")
                st.caption("Comportamento médio da frequência dos estudantes de segunda a sexta-feira.")
                
                chart_d_bar = alt.Chart(df_d).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#008744").encode(
                    x=alt.X("dia_nome:N", sort=["Segunda", "Terça", "Quarta", "Quinta", "Sexta"], title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("consumo_medio:Q", title="Média de Refeições Servidas"),
                    tooltip=[
                        alt.Tooltip("dia_nome:N", title="Dia da Semana"),
                        alt.Tooltip("consumo_medio:Q", title="Média (refeições)"),
                        alt.Tooltip("count:Q", title="Total de Dias")
                    ]
                ).properties(height=260).interactive()
                st.altair_chart(chart_d_bar, use_container_width=True)
                
            with col_d2:
                st.markdown("##### Sazonalidade ao Longo dos Meses do Ano Letivo")
                st.caption("Picos no meio dos semestres (março/agosto) e quedas típicas em inícios de ano letivo e períodos de recesso.")
                
                df_dash_mes = df_dash.copy()
                df_dash_mes["mes_nome"] = df_dash_mes["mes"].map(meses_map)
                df_m = df_dash_mes.groupby(["mes", "mes_nome"])["consumo"].mean().reset_index().sort_values("mes")
                df_m["consumo_medio"] = df_m["consumo"].round(1)
                
                chart_m_bar = alt.Chart(df_m).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#0284c7").encode(
                    x=alt.X("mes_nome:N", sort=list(meses_map.values()), title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("consumo_medio:Q", title="Média de Refeições"),
                    tooltip=[
                        alt.Tooltip("mes_nome:N", title="Mês"),
                        alt.Tooltip("consumo_medio:Q", title="Média (ref)")
                    ]
                ).properties(height=260).interactive()
                st.altair_chart(chart_m_bar, use_container_width=True)

            # Linha 2: Desmobilização de Fim de Semestre e Curva Semanal por Ano
            col_d3, col_d4 = st.columns(2)
            with col_d3:
                st.markdown("##### Desmobilização Acadêmica no Fim do Semestre")
                st.caption("Queda expressiva de consumo nos últimos 15 dias letivos antes das férias (Dissertação Fig. 05).")
                
                df_fim = df_validos_geral.dropna(subset=["dias_ate_fim_bloco"]).copy()
                df_fim = df_fim[df_fim["dias_ate_fim_bloco"] <= 20]
                df_fim_agg = df_fim.groupby("dias_ate_fim_bloco")["consumo"].agg(["mean", "count"]).reset_index()
                df_fim_agg["dias_restantes"] = df_fim_agg["dias_ate_fim_bloco"].astype(int)
                
                chart_fim = alt.Chart(df_fim_agg).mark_line(point=True, color="#dc2626", strokeWidth=2.5).encode(
                    x=alt.X("dias_restantes:Q", sort="descending", title="Dias Restantes até o Término das Aulas"),
                    y=alt.Y("mean:Q", title="Consumo Médio de Refeições"),
                    tooltip=[
                        alt.Tooltip("dias_restantes:Q", title="Dias até o Fim"),
                        alt.Tooltip("mean:Q", title="Consumo Médio (ref)", format=".1f"),
                        alt.Tooltip("count:Q", title="Amostras no Histórico")
                    ]
                ).properties(height=260).interactive()
                st.altair_chart(chart_fim, use_container_width=True)

            with col_d4:
                st.markdown("##### Curva Semanal Comparativa por Ano (2023–2026)")
                st.caption("Comprova que a queda de sexta-feira é estrutural e se repetiu em todos os 4 anos analisados.")
                
                df_sem_ano = df_validos_geral[df_validos_geral["dia_da_semana"].isin(dias_ordem)].groupby(["ano", "dia_da_semana"])["consumo"].mean().reset_index()
                df_sem_ano["dia_nome"] = df_sem_ano["dia_da_semana"].map(dias_map)
                df_sem_ano["ano_str"] = df_sem_ano["ano"].astype(str)
                
                chart_sem_ano = alt.Chart(df_sem_ano).mark_line(point=True, strokeWidth=2.2).encode(
                    x=alt.X("dia_nome:N", sort=["Segunda", "Terça", "Quarta", "Quinta", "Sexta"], title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("consumo:Q", title="Média de Refeições"),
                    color=alt.Color("ano_str:N", title="Ano", scale=alt.Scale(scheme="category10"), legend=alt.Legend(orient="top")),
                    tooltip=[
                        alt.Tooltip("ano_str:N", title="Ano"),
                        alt.Tooltip("dia_nome:N", title="Dia"),
                        alt.Tooltip("consumo:Q", title="Média (ref)", format=".0f")
                    ]
                ).properties(height=260).interactive()
                st.altair_chart(chart_sem_ano, use_container_width=True)

            with st.expander("🖼️ Ver Figuras Acadêmicas da Dissertação (Figuras 03, 04 e 05)", expanded=False):
                col_fa, col_fb, col_fc = st.columns(3)
                with col_fa:
                    p3 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_03_perfil_semanal_por_ano.png")
                    if os.path.exists(p3):
                        st.image(p3, caption="Figura 03: Perfil Semanal Facetado por Ano", use_container_width=True)
                with col_fb:
                    p4 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_04_sazonalidade_mensal.png")
                    if os.path.exists(p4):
                        st.image(p4, caption="Figura 04: Sazonalidade Mensal do Consumo", use_container_width=True)
                with col_fc:
                    p5 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_05_desmobilizacao_fim_semestre.png")
                    if os.path.exists(p5):
                        st.image(p5, caption="Figura 05: Desmobilização no Término do Semestre", use_container_width=True)

    # --------------------------------------------------------------------------
    # SUBSEÇÃO 3: CARDÁPIO & ATRATIVIDADE
    # --------------------------------------------------------------------------
    elif "3. Cardápio" in secao_dash:
        with st.container(border=True):
            st.markdown("#### 🍲 Qual Prato Atrai Mais Estudantes?")
            st.write(
                "**A preferência discente no prato principal:** "
                "O tipo de proteína é uma das variáveis mais fortes no modelo. "
                "Pratos à base de carne bovina e frango registram alta adesão, enquanto pratos com peixe historicamente têm menor procura."
            )
            
            nomes_proteina = {
                "bovina": "Carne Bovina",
                "frango": "Frango / Aves",
                "suina": "Carne Suína",
                "peixe": "Peixe / Pescados"
            }
            
            df_p = df_dash[df_dash["proteina_tipo"].isin(nomes_proteina.keys())].copy()
            df_p_agg = df_p.groupby("proteina_tipo")["consumo"].agg(["count", "mean", "min", "max"]).reset_index()
            df_p_agg["proteina_nome"] = df_p_agg["proteina_tipo"].map(nomes_proteina)
            df_p_agg["consumo_medio"] = df_p_agg["mean"].round(1)
            df_p_agg = df_p_agg.sort_values(by="consumo_medio", ascending=False).reset_index(drop=True)
            
            col_p1, col_p2 = st.columns([1, 1.3])
            with col_p1:
                st.markdown("##### Média de Alunos por Tipo de Carne")
                st.caption(f"Comparativo visual de consumo médio no período: **{filtro_ano}**.")
                
                chart_p_bar = alt.Chart(df_p_agg).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#008744").encode(
                    x=alt.X("proteina_nome:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("consumo_medio:Q", title="Média de Refeições Servidas"),
                    tooltip=[
                        alt.Tooltip("proteina_nome:N", title="Proteína"),
                        alt.Tooltip("consumo_medio:Q", title="Média (refeições)"),
                        alt.Tooltip("count:Q", title="Dias Servidos")
                    ]
                ).properties(height=280).interactive()
                st.altair_chart(chart_p_bar, use_container_width=True)
                
            with col_p2:
                st.markdown(f"##### Ranking e Recomendações por Prato — {filtro_ano}")
                st.caption(f"Estatísticas dinâmicas calculadas em tempo real sobre {len(df_p)} datas analisadas no filtro.")
                
                html_rows = []
                rank_meta = [
                    ("1º Lugar", "rank-1", "🏆 Maior Procura"),
                    ("2º Lugar", "rank-2", "⭐ Alta Procura"),
                    ("3º Lugar", "rank-3", "⭐ Procura Média"),
                    ("4º Lugar", "rank-other", "⭐⭐ Procura Moderada")
                ]
                
                for idx, r_p in df_p_agg.iterrows():
                    p_nome = r_p["proteina_nome"]
                    cnt = int(r_p["count"])
                    med = float(r_p["mean"])
                    v_min = int(r_p["min"])
                    v_max = int(r_p["max"])
                    
                    b_txt, b_cls, b_stars = rank_meta[idx] if idx < len(rank_meta) else (f"{idx+1}º Lugar", "rank-other", "⭐ Regular")
                    
                    if idx == 0:
                        acao = "Maior demanda apurada. Manter quantitativo pleno e margem protetiva total."
                    elif idx == 1:
                        acao = "Padrão de comparecimento consolidado. Manter produção regular estável."
                    elif idx == 2:
                        acao = "Procura intermediária. Acompanhar sobras de balcão e dispensar reforço."
                    else:
                        acao = "Menor adesão no recorte. Reduzir preparo em ~15 a 20 ref para evitar sobras."
                        
                    row_html = (
                        '<tr style="border-bottom: 1px solid var(--border-subtle);">'
                        f'<td style="padding: 9px 10px; font-weight: 700; white-space: nowrap;"><span class="rank-badge {b_cls}">{b_txt}</span></td>'
                        f'<td style="padding: 9px 10px; font-weight: 600; color: var(--text-main);">{html.escape(p_nome)}<br><small style="color: var(--text-muted); font-size: 11px;">{b_stars}</small></td>'
                        f'<td style="padding: 9px 10px; font-weight: 800; font-family: var(--font-mono); white-space: nowrap; color: var(--text-main);">{med:.0f} ref<br><small style="color: var(--text-muted); font-family: var(--font-sans); font-size: 11px; font-weight: normal;">mín {v_min} / máx {v_max}</small></td>'
                        f'<td style="padding: 9px 10px; text-align: center; font-family: var(--font-mono); font-size: 12px; white-space: nowrap; color: var(--text-muted);">{cnt} datas</td>'
                        f'<td style="padding: 9px 10px; font-size: 12px; line-height: 1.35; color: var(--text-main);">{acao}</td>'
                        '</tr>'
                    )
                    html_rows.append(row_html)
                    
                table_html = (
                    '<div style="overflow-x: auto; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); background: var(--bg-card); margin-bottom: 8px;">'
                    '<table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 12.5px;">'
                    '<thead><tr style="background: var(--bg-secondary); border-bottom: 2px solid var(--border-subtle);">'
                    '<th style="padding: 9px 10px; color: var(--text-main); font-weight: 700;">Posição</th>'
                    '<th style="padding: 9px 10px; color: var(--text-main); font-weight: 700;">Proteína</th>'
                    '<th style="padding: 9px 10px; color: var(--text-main); font-weight: 700;">Média no Filtro</th>'
                    '<th style="padding: 9px 10px; text-align: center; color: var(--text-main); font-weight: 700;">Datas</th>'
                    '<th style="padding: 9px 10px; color: var(--text-main); font-weight: 700;">Recomendação para a Cozinha</th>'
                    '</tr></thead><tbody>' + "".join(html_rows) + '</tbody></table></div>'
                )
                if hasattr(st, "html"):
                    st.html(table_html)
                else:
                    st.markdown(table_html, unsafe_allow_html=True)

            # Linha 2: Técnica Culinária e Tabela de Variabilidade/Risco Operacional
            col_p3, col_p4 = st.columns(2)
            with col_p3:
                st.markdown("##### Adesão por Técnica Culinária / Tipo de Preparo")
                st.caption("Frequência média discente conforme a técnica de cocção (Dissertação Fig. 07).")
                
                preparo_map = {
                    "ao_molho_ou_refogado": "Ao Molho / Refogado",
                    "assado": "Assado ao Forno",
                    "cozido_ou_refogado": "Cozido / Ensopado",
                    "cremoso_ou_ensopado": "Cremoso / Estrogonofe",
                    "massa_ou_montado": "Massas / Pratos Montados",
                    "grelhado": "Grelhado na Chapa",
                    "empanado_ou_a_milanesa": "Empanado / À Milanesa",
                    "churrasco_ou_espeto": "Churrasco / Espeto",
                    "frito": "Frito / Dourado"
                }
                df_prep = df_dash[df_dash["preparo_tipo"].isin(preparo_map.keys())].copy()
                df_prep["preparo_nome"] = df_prep["preparo_tipo"].map(preparo_map)
                df_prep_agg = df_prep.groupby("preparo_nome")["consumo"].agg(["count", "mean"]).reset_index()
                df_prep_agg["consumo_medio"] = df_prep_agg["mean"].round(1)
                df_prep_agg = df_prep_agg.sort_values(by="consumo_medio", ascending=False)
                
                chart_prep_bar = alt.Chart(df_prep_agg).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#0284c7").encode(
                    x=alt.X("preparo_nome:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("consumo_medio:Q", title="Média de Refeições"),
                    tooltip=[
                        alt.Tooltip("preparo_nome:N", title="Técnica Culinária"),
                        alt.Tooltip("consumo_medio:Q", title="Média de Refeições"),
                        alt.Tooltip("count:Q", title="Datas Servidas")
                    ]
                ).properties(height=260).interactive()
                st.altair_chart(chart_prep_bar, use_container_width=True)

            with col_p4:
                st.markdown("##### Variabilidade & Risco Operacional por Carne")
                st.caption("Piso, média, teto e amplitude de variação da demanda por carne no período.")
                
                tot_p = len(df_p)
                resumo_var = []
                for p_k, p_n in nomes_proteina.items():
                    sub_p = df_p[df_p["proteina_tipo"] == p_k]
                    cnt_p = len(sub_p)
                    if cnt_p > 0:
                        pct_f = (cnt_p / tot_p) * 100 if tot_p > 0 else 0
                        v_mi = sub_p["consumo"].min()
                        v_me = sub_p["consumo"].mean()
                        v_ma = sub_p["consumo"].max()
                        amp = v_ma - v_mi
                        diag = "Baixo Risco (Demanda volumosa e estável)" if p_k in ["bovina", "frango"] else ("Médio Risco (Intermediária)" if p_k == "suina" else "Atenção (Menor média e alta sensibilidade)")
                        resumo_var.append({
                            "Proteína": p_n,
                            "Frequência": f"{pct_f:.1f}% ({cnt_p}d)",
                            "Mín": f"{int(v_mi)} ref",
                            "Média": f"{v_me:.0f} ref",
                            "Máx": f"{int(v_ma)} ref",
                            "Oscilação": f"±{amp:.0f} ref",
                            "Classificação Operacional": diag
                        })
                df_tab_var = pd.DataFrame(resumo_var)
                st.dataframe(df_tab_var, use_container_width=True, hide_index=True)

            with st.expander("🖼️ Ver Figuras Acadêmicas da Dissertação (Figuras 06 e 07)", expanded=False):
                col_f_c1, col_f_c2 = st.columns(2)
                with col_f_c1:
                    p6 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_06_consumo_por_proteina.png")
                    if os.path.exists(p6):
                        st.image(p6, caption="Figura 06: Demanda e Frequência por Tipo de Proteína", use_container_width=True)
                with col_f_c2:
                    p7 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_07_consumo_por_metodo_preparo.png")
                    if os.path.exists(p7):
                        st.image(p7, caption="Figura 07: Consumo Conforme a Técnica Culinária de Preparo", use_container_width=True)

    # --------------------------------------------------------------------------
    # SUBSEÇÃO 4: CLIMA & EFEITO DAS CHUVAS
    # --------------------------------------------------------------------------
    elif "4. Clima" in secao_dash:
        with st.container(border=True):
            st.markdown("#### 🌦️ Como o Clima Afeta a Presença Discente?")
            st.write(
                "**Desmistificando a chuva:** "
                "A pesquisa revelou que garoas e chuvas leves quase não alteram a frequência dos estudantes no refeitório. "
                "O impacto negativo real ocorre exclusivamente em **chuvas fortes e tempestades (>15 mm)**, "
                "que provocam inundações em vias e na linha da CPTM, impedindo os alunos de chegarem ao campus."
            )
            
            df_ch = df_dash.copy()
            df_ch["faixa_chuva"] = pd.cut(
                df_ch["precipitation_sum"],
                bins=[-1, 0.1, 5, 15, 100],
                labels=["Dia Seco (0 mm)", "Garoa (0.1 a 5 mm)", "Chuva Forte (5.1 a 15 mm)", "Tempestade (>15 mm)"]
            )
            df_ch_agg = df_ch.groupby("faixa_chuva", observed=False)["consumo"].agg(["count", "mean"]).reset_index()
            df_ch_agg["consumo_medio"] = df_ch_agg["mean"].round(1)
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("##### Consumo Médio por Intensidade de Chuva")
                st.caption("Queda acentuada de alunos nos dias de tempestade (>15 mm).")
                
                chart_ch_bar = alt.Chart(df_ch_agg).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#0284c7").encode(
                    x=alt.X("faixa_chuva:N", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("consumo_medio:Q", title="Média de Refeições Servidas"),
                    tooltip=[
                        alt.Tooltip("faixa_chuva:N", title="Condição Climática"),
                        alt.Tooltip("consumo_medio:Q", title="Média de Refeições"),
                        alt.Tooltip("count:Q", title="Total de Dias no Histórico")
                    ]
                ).properties(height=260).interactive()
                st.altair_chart(chart_ch_bar, use_container_width=True)
                
            with col_c2:
                st.markdown("##### Relação Chuva (mm) vs. Refeições Servidas")
                st.caption("Cada ponto representa um dia letivo real; a linha vermelha mostra a tendência.")
                
                chart_ch_scatter = alt.Chart(df_ch).mark_circle(size=45, opacity=0.6, color="#0284c7").encode(
                    x=alt.X("precipitation_sum:Q", title="Volume de Chuva (mm)"),
                    y=alt.Y("consumo:Q", title="Refeições Servidas"),
                    tooltip=[
                        alt.Tooltip("data:N", title="Data"),
                        alt.Tooltip("precipitation_sum:Q", title="Chuva (mm)"),
                        alt.Tooltip("consumo:Q", title="Consumo Real"),
                        alt.Tooltip("prato_principal:N", title="Prato")
                    ]
                )
                line_trend = chart_ch_scatter.transform_regression("precipitation_sum", "consumo").mark_line(color="#ef4444", strokeWidth=2.5)
                st.altair_chart((chart_ch_scatter + line_trend).properties(height=260).interactive(), use_container_width=True)

            col_c3, col_c4 = st.columns(2)
            with col_c3:
                st.markdown("##### Frequência Discente por Faixa de Temperatura")
                st.caption("Adesão de estudantes conforme a temperatura máxima registrada.")
                
                df_ch["faixa_temp"] = pd.cut(
                    df_ch["temperature_2m_max"],
                    bins=[-10, 20, 25, 30, 50],
                    labels=["Frio (<20°C)", "Agradável (20 a 25°C)", "Calor Moderado (25 a 30°C)", "Calor Intenso (>30°C)"]
                )
                df_t_agg = df_ch.groupby("faixa_temp", observed=False)["consumo"].agg(["count", "mean"]).reset_index()
                df_t_agg["consumo_medio"] = df_t_agg["mean"].round(1)
                
                chart_temp_bar = alt.Chart(df_t_agg).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#f59e0b").encode(
                    x=alt.X("faixa_temp:N", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("consumo_medio:Q", title="Média de Refeições"),
                    tooltip=[
                        alt.Tooltip("faixa_temp:N", title="Faixa Térmica"),
                        alt.Tooltip("consumo_medio:Q", title="Média de Refeições"),
                        alt.Tooltip("count:Q", title="Total de Dias")
                    ]
                ).properties(height=260).interactive()
                st.altair_chart(chart_temp_bar, use_container_width=True)
                
            with col_c4:
                st.markdown("##### Matriz de Diagnóstico Meteorológico & Ação")
                st.caption("Regras de calibração automática aplicadas pelo artefato preditivo.")
                
                d_sec = (df_ch["precipitation_sum"] == 0).sum()
                m_sec = df_ch[df_ch["precipitation_sum"] == 0]["consumo"].mean()
                d_gar = ((df_ch["precipitation_sum"] > 0) & (df_ch["precipitation_sum"] <= 5)).sum()
                m_gar = df_ch[(df_ch["precipitation_sum"] > 0) & (df_ch["precipitation_sum"] <= 5)]["consumo"].mean()
                d_chu = ((df_ch["precipitation_sum"] > 5) & (df_ch["precipitation_sum"] <= 15)).sum()
                m_chu = df_ch[(df_ch["precipitation_sum"] > 5) & (df_ch["precipitation_sum"] <= 15)]["consumo"].mean()
                d_tem = (df_ch["precipitation_sum"] > 15).sum()
                m_tem = df_ch[df_ch["precipitation_sum"] > 15]["consumo"].mean()
                
                df_matriz_clima = pd.DataFrame([
                    {"Condição Climática": "Dia Seco (0 mm)", "Datas": f"{d_sec}d", "Média Real": f"{m_sec:.0f} ref" if pd.notna(m_sec) else "-", "Regra do Artefato": "Margem protetiva plena (+12 ref)."},
                    {"Condição Climática": "Garoa (0.1 a 5 mm)", "Datas": f"{d_gar}d", "Média Real": f"{m_gar:.0f} ref" if pd.notna(m_gar) else "-", "Regra do Artefato": "Pequeno recuo automático (-5 ref)."},
                    {"Condição Climática": "Chuva Forte (5.1 a 15 mm)", "Datas": f"{d_chu}d", "Média Real": f"{m_chu:.0f} ref" if pd.notna(m_chu) else "-", "Regra do Artefato": "Recuo preventivo (-12 ref)."},
                    {"Condição Climática": "Tempestade (>15 mm)", "Datas": f"{d_tem}d", "Média Real": f"{m_tem:.0f} ref" if pd.notna(m_tem) else "-", "Regra do Artefato": "Zera margem extra e corta excesso."}
                ])
                st.dataframe(df_matriz_clima, use_container_width=True, hide_index=True)

            with st.expander("🖼️ Ver Figuras Acadêmicas da Dissertação (Figuras 08 e 09)", expanded=False):
                col_f_cl1, col_f_cl2 = st.columns(2)
                with col_f_cl1:
                    p8 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_08_sensibilidade_clima_precipitacao.png")
                    if os.path.exists(p8):
                        st.image(p8, caption="Figura 08: Sensibilidade Climática por Faixas de Precipitação", use_container_width=True)
                with col_f_cl2:
                    p9 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_09_relacao_temperatura_consumo.png")
                    if os.path.exists(p9):
                        st.image(p9, caption="Figura 09: Relação Térmica entre Temperatura Máxima e Consumo", use_container_width=True)

    # --------------------------------------------------------------------------
    # SUBSEÇÃO 5: PODER PREDITIVO DOS ATRIBUTOS (EDA & MULTIVARIADA)
    # --------------------------------------------------------------------------
    elif "5. Poder Preditivo" in secao_dash:
        with st.container(border=True):
            st.markdown("#### 🌪️ Ranking de Importância e Poder Preditivo dos Atributos")
            st.write(
                "**Por que usamos cada variável nos modelos preditivos?** "
                "Para provar cientificamente à banca examinadora quais fatores realmente governam o refeitório, "
                "analisamos a correlação de Pearson de todos os atributos. O histórico recente e o dia da semana explicam a maior parte da demanda, "
                "enquanto a chuva comum tem impacto moderado e dias de encerramento acadêmico provocam quedas acentuadas."
            )
            
            # Tornado Chart Interativo dos Atributos
            dados_tornado = [
                {"Atributo": "Nível Basal Recente (Mediana 10d)", "Correlacao": 0.658, "Tipo": "Histórico / Inércia", "Impacto": "Positivo Forte"},
                {"Atributo": "Inércia Mesmo Dia da Semana", "Correlacao": 0.656, "Tipo": "Calendário / Ciclo", "Impacto": "Positivo Forte"},
                {"Atributo": "Média Móvel Exponencial (EWMA)", "Correlacao": 0.642, "Tipo": "Histórico / Inércia", "Impacto": "Positivo Forte"},
                {"Atributo": "Dias até o Fim do Semestre", "Correlacao": 0.334, "Tipo": "Calendário / Prazo", "Impacto": "Positivo Moderado"},
                {"Atributo": "Cardápio: Carne Bovina", "Correlacao": 0.285, "Tipo": "Cardápio / Atratividade", "Impacto": "Positivo Moderado"},
                {"Atributo": "Dias Decorridos desde Início", "Correlacao": 0.162, "Tipo": "Calendário / Fase", "Impacto": "Positivo Leve"},
                {"Atributo": "Cardápio: Carne Suína", "Correlacao": 0.082, "Tipo": "Cardápio / Atratividade", "Impacto": "Positivo Leve"},
                {"Atributo": "Previsão de Chuva (mm)", "Correlacao": 0.041, "Tipo": "Meteorologia", "Impacto": "Neutro / Residual"},
                {"Atributo": "Cardápio: Aves / Frango", "Correlacao": 0.035, "Tipo": "Cardápio / Atratividade", "Impacto": "Neutro / Padrão"},
                {"Atributo": "Previsão Temperatura Máxima", "Correlacao": -0.029, "Tipo": "Meteorologia", "Impacto": "Neutro / Residual"},
                {"Atributo": "Preparo: Grelhado / Chapa", "Correlacao": -0.098, "Tipo": "Cardápio / Cocção", "Impacto": "Negativo Leve"},
                {"Atributo": "Cardápio: Peixe / Pescados", "Correlacao": -0.198, "Tipo": "Cardápio / Rejeição", "Impacto": "Negativo Moderado"},
                {"Atributo": "Dia da Semana: Sexta-feira", "Correlacao": -0.215, "Tipo": "Calendário / Evasão", "Impacto": "Negativo Moderado"},
                {"Atributo": "Últimos 3 Dias de Semestre", "Correlacao": -0.380, "Tipo": "Calendário / Desmobilização", "Impacto": "Negativo Severo"}
            ]
            df_tor = pd.DataFrame(dados_tornado)
            
            col_t1, col_t2 = st.columns([1.2, 0.8])
            with col_t1:
                st.markdown("##### Tornado Chart: Correlação dos Atributos com o Consumo Real")
                st.caption("Verde indica correlação positiva (aumenta o consumo); Vermelho indica efeito redutor.")
                
                chart_tornado = alt.Chart(df_tor).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                    y=alt.Y("Atributo:N", sort="-x", title=None),
                    x=alt.X("Correlacao:Q", title="Correlação Linear de Pearson (r)"),
                    color=alt.condition(
                        alt.datum.Correlacao > 0,
                        alt.value("#008744"),
                        alt.value("#ef4444")
                    ),
                    tooltip=[
                        alt.Tooltip("Atributo:N", title="Atributo"),
                        alt.Tooltip("Correlacao:Q", title="Correlação (r)", format="+.3f"),
                        alt.Tooltip("Tipo:N", title="Dimensão"),
                        alt.Tooltip("Impacto:N", title="Efeito no Consumo")
                    ]
                ).properties(height=380).interactive()
                st.altair_chart(chart_tornado, use_container_width=True)
                
            with col_t2:
                st.markdown("##### Autocorrelação Serial (Ciclos Semanais)")
                st.caption("Picos periódicos nos lags 5, 10 e 15 provam a forte sazonalidade semanal de 5 dias úteis.")
                
                # Autocorrelação real
                lags = list(range(1, 21))
                autocorr_vals = []
                s_cons = df_validos_geral["consumo"].dropna()
                for lg in lags:
                    autocorr_vals.append(float(s_cons.autocorr(lag=lg)))
                    
                df_autocorr = pd.DataFrame({
                    "Lag": lags,
                    "Autocorrelacao": autocorr_vals,
                    "Destaque": ["Pico Semanal (5d)" if l in [5, 10, 15] else "Lag Ordinário" for l in lags]
                })
                
                chart_auto = alt.Chart(df_autocorr).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X("Lag:O", title="Defasagem Temporal (Dias Úteis)"),
                    y=alt.Y("Autocorrelacao:Q", title="Autocorrelação"),
                    color=alt.Color("Destaque:N", scale=alt.Scale(domain=["Pico Semanal (5d)", "Lag Ordinário"], range=["#008744", "#94a3b8"]), legend=alt.Legend(orient="top", title=None)),
                    tooltip=[
                        alt.Tooltip("Lag:O", title="Lag (Dias Úteis)"),
                        alt.Tooltip("Autocorrelacao:Q", title="Autocorrelação", format=".3f"),
                        alt.Tooltip("Destaque:N", title="Classificação")
                    ]
                ).properties(height=380).interactive()
                st.altair_chart(chart_auto, use_container_width=True)

            with st.expander("🖼️ Ver Figuras Acadêmicas da Dissertação (Figuras 10, 11 e 12)", expanded=False):
                col_f_e1, col_f_e2, col_f_e3 = st.columns(3)
                with col_f_e1:
                    p10 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_10_matriz_correlacao_multivariada.png")
                    if os.path.exists(p10):
                        st.image(p10, caption="Figura 10: Matriz de Correlação Multivariada", use_container_width=True)
                with col_f_e2:
                    p11 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_11_autocorrelacao_serial.png")
                    if os.path.exists(p11):
                        st.image(p11, caption="Figura 11: Autocorrelação Serial (Lags 1 a 20)", use_container_width=True)
                with col_f_e3:
                    p12 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_12_tornado_importancia_atributos.png")
                    if os.path.exists(p12):
                        st.image(p12, caption="Figura 12: Tornado Chart de Poder Preditivo", use_container_width=True)

'''

conteudo_final = parte_antes + novo_dashboard + parte_depois
app_path.write_text(conteudo_final, encoding="utf-8")
print("Reconstrução da Página 1 (Dashboard) concluída com sucesso!")
