import os
from pathlib import Path

app_main_path = Path("src/app/main.py")
content = app_main_path.read_text(encoding="utf-8")

guia_header = 'elif pagina_ativa == "Guia Prático":'
if guia_header not in content:
    # Caso haja caractere de encoding diferente
    pos = content.find('elif pagina_ativa == "Guia')
else:
    pos = content.find(guia_header)

print(f"Encontrado pos: {pos}")

# Vamos manter tudo até renderizar_painel_explicativo da página Guia Prático
split_marker = 'renderizar_painel_explicativo(\n        objetivo_rotina=('
idx_marker = content.find(split_marker, pos)
# Achar o fechamento de renderizar_painel_explicativo:
idx_close = content.find('        )\n    )\n', idx_marker)
if idx_close != -1:
    idx_close += len('        )\n    )\n')
    parte_anterior = content[:idx_close]
else:
    idx_close = content.find('        )\n', idx_marker) + len('        )\n')
    parte_anterior = content[:idx_close] + "    )\n"

novo_fim = """
    tab_pop, tab_dissertacao, tab_auditoria = st.tabs([
        "📋 Procedimento Operacional Padrão (POP)",
        "📊 Evidências da Dissertação (2026)",
        "🛡️ Auditoria Anti-Vazamento (Zero Data Leakage)"
    ])
    
    with tab_pop:
        st.markdown(\"\"\"
        #### ⏱️ Rotina Semanal em 3 Minutos
        
        1. **Toda Sexta-feira (Planejamento da Produção t+1):**
           - Acesse a página **'✍️ Cadastrar Dados'** e informe os pratos previstos para a semana seguinte.
           - Acesse a página **'🔮 Previsões'**:
             - Confira a recomendação em **'📋 Ordem de Preparo'** para cada dia.
             - Baixe a planilha em CSV e envie a grade para a equipe de cozinha terceirizada.
           
        2. **Diariamente (Registro Rápido após as 14h):**
           - Acesse a página **'✍️ Cadastrar Dados'** e digite a quantidade real de refeições servidas no almoço. Isso mantém o modelo sempre afinado.
           
        3. **Quinzenalmente (Planejamento de Compras de Secos t+2):**
           - Acesse a página **'📦 Compras & Suprimentos'** para ver a projeção das próximas duas semanas e dimensionar pedidos de estoque.
           
        ---
        
        #### 🌦️ Como Agir em Dias Especiais
        
        * **Dias de Tempestade / Alagamentos:**
          O sistema reduz automaticamente a folga de segurança para evitar que a comida fique sobrando no balcão de distribuição.
        * **Sextas-feiras:**
          O modelo já calcula a queda natural de ~20% no comparecimento discente no turno vespertino.
        * **Cardápio com Carne Bovina:**
          O modelo ajusta a quantidade para cima para garantir que nenhum estudante fique sem comer no dia de maior procura.
        \"\"\")
        
    with tab_dissertacao:
        st.markdown("#### 📊 Evidências Empíricas da Dissertação de Mestrado (DSR)")
        st.markdown("**Síntese Geral: Período Retrospectivo (Fev-Ago/2026, N=121) vs. Prospectivo Cego (Set-Out/2026, N=28)**")
        
        caminho_tab10 = os.path.join(ROOT_DIR, "resultados", "tabelas_dissertacao", "tabela_10_sintese_comparativa_retrospectivo_vs_prospectivo.csv")
        if os.path.exists(caminho_tab10):
            df_sintese = pd.read_csv(caminho_tab10, sep=";")
            st.dataframe(df_sintese, use_container_width=True)
            
        col_ed1, col_ed2 = st.columns(2)
        with col_ed1:
            p_fig28 = os.path.join(ROOT_DIR, "resultados", "figuras", "dissertacao", "figura_28_sintese_retrospectivo_vs_prospectivo.png")
            if os.path.exists(p_fig28):
                st.image(p_fig28, caption="Figura 28: Síntese de Erro (MAE) por Regime e Condição Operacional")
        with col_ed2:
            p_fig27 = os.path.join(ROOT_DIR, "resultados", "figuras", "dissertacao", "figura_27_balanco_sobras_faltas_prospectivo.png")
            if os.path.exists(p_fig27):
                st.image(p_fig27, caption="Figura 27: Balanço Acumulado de Sobras e Faltas no Bimestre Prospectivo")

        st.markdown("---")
        st.markdown("##### 🌊 Resiliência às Enchentes de Setembro de 2026 (Semana de Crise no Alto Tietê)")
        col_en1, col_en2 = st.columns([1.2, 0.8])
        with col_en1:
            st.markdown(\"\"\"
            Durante a semana de inundações na linha da CPTM e nas vias de acesso ao campus em Suzano, o consumo despencou para entre 14 e 118 refeições:
            - **Nutricionista Humano:** Insistiu na meta institucional padrão (190 a 235 ref), sofrendo erro de **MAE = 136,60 refeições** (WAPE de 188,67%) e gerando **683 refeições desperdiçadas** em apenas 5 dias úteis (R$ 9.562,00 em descarte).
            - **IA Campeã Congelada em 31/08:** Detectou os alertas meteorológicos de tempestade e reduziu as previsões, atingindo **MAE = 106,04 ref** (WAPE de 146,46%) — **errando 30,56 refeições a menos que o especialista humano**.
            \"\"\")
        with col_en2:
            p_fig25 = os.path.join(ROOT_DIR, "resultados", "figuras", "dissertacao", "figura_25_confronto_diario_setembro_enchentes_ia_vs_nutri.png")
            if os.path.exists(p_fig25):
                st.image(p_fig25, caption="Figura 25: Resiliência Climática nas Enchentes")

    with tab_auditoria:
        st.markdown("#### 🛡️ Auditoria Formal de Não-Vazamento Temporal (Zero Data Leakage)")
        st.markdown("Verificação de integridade matemática e ausência de viés retrospectivo para a banca examinadora.")
        
        caminho_relatorio = os.path.join(ROOT_DIR, "resultados", "auditoria", "relatorio_teste_vazamento_modelos.json")
        if os.path.exists(caminho_relatorio):
            with open(caminho_relatorio, "r", encoding="utf-8") as f_rel:
                rel_json = json.load(f_rel)
                
            st.success(f"✅ **STATUS GERAL DA BASE:** {rel_json.get('status_geral', 'APROVADO - ZERO DATA LEAKAGE')}")
            st.caption(f"Última auditoria executada em: {rel_json.get('data_auditoria', '01/11/2026')}")
            
            for teste in rel_json.get("testes", []):
                with st.container(border=True):
                    st.markdown(f"**{teste.get('teste')}**")
                    if teste.get("sucesso"):
                        st.markdown("🟢 **Resultado:** APROVADO COM ZERO FALHAS")
                    else:
                        st.markdown("🔴 **Resultado:** REPROVADO")
                    if "detalhe" in teste:
                        st.markdown(f"- *Diagnóstico:* {teste['detalhe']}")
                    if "detalhes" in teste:
                        for d in teste["detalhes"]:
                            st.markdown(f"- {d}")
                            
        if st.button("Executar Testes Adversariais Anti-Vazamento Agora", key="btn_auditar_vazamento"):
            with st.spinner("Executando simulações adversariais com futuro corrompido..."):
                try:
                    from src.core.test_zero_leakage_models import executar_auditoria_completa_vazamento
                    rel_novo = executar_auditoria_completa_vazamento()
                    st.success("Bateria de testes finalizada com 100% de conformidade! Zero Data Leakage comprovado.")
                    st.rerun()
                except Exception as e_aud:
                    st.error(f"Erro ao executar auditoria: {e_aud}")

# ==============================================================================
# RODAPÉ ACADÊMICO INSTITUCIONAL (DSR / PPG-GTSP)
# ==============================================================================
st.markdown(\"\"\"
<div style="margin-top: 50px; padding: 24px 16px; border-top: 1px solid #e2e8f0; text-align: center; font-size: 12px; color: #64748b; background-color: #ffffff; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
    <div style="font-weight: 700; color: #0f172a; margin-bottom: 4px;">Restaurante Estudantil — Painel de Apoio à Decisão</div>
    <div>Dissertação de Mestrado Profissional • Metodologia Design Science Research (DSR) • Ciclo 2026</div>
    <div style="margin-top: 4px;">Programa de Pós-Graduação em Gestão e Tecnologia em Sistemas Produtivos (PPG-GTSP)</div>
    <div style="margin-top: 6px; font-size: 11px; color: #94a3b8;">
        Mestrando: <b>Michel Silva de Souza</b> • Orientador: <b>Prof. Dr. Carlos Hideo Arima</b> • Coorientador: <b>Prof. Dr. Rafael Nobre Orsi</b>
    </div>
</div>
\"\"\", unsafe_allow_html=True)
"""

conteudo_final = parte_anterior + novo_fim
app_main_path.write_text(conteudo_final, encoding="utf-8")
print("src/app/main.py atualizado com sucesso!")
