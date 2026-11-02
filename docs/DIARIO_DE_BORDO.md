# Diário de Bordo do Projeto

**Artefato Analítico-Preditivo para Planejamento de Demanda e Mitigação do Desperdício em Restaurante Institucional de Ensino**  
*Mestrado Profissional em Educação Profissional e Tecnológica (ProfEPT) / Gestão Pública*  
**Pesquisador:** Michel Silva de Souza  
**Metodologia Base:** *Design Science Research* (DSR)  
**Repositório Oficial:** `F:\PROJETOS\ARTEFATO_ANALITICO_PREDITIVO`  

---

## 📌 Compromissos e Diretrizes Metodológicas
1. **Zero Data Leakage (Isolamento Temporal Absoluto):** Nenhuma informação futura (incluindo consumos de dias da própria semana planejada ou médias móveis globais) pode contaminar as decisões preditivas emitidas na sexta-feira de corte.
2. **Utilidade Operacional Real:** O artefato atende a dois problemas concretos da escola pública: o **preparo diário ($T+1$)** e as **compras de insumos ($T+2$)**.
3. **Transparência e Reprodutibilidade:** Toda transformação, classificação de outliers e seleção de hiperparâmetros é determinística, auditável e livre de viés retrospectivo.
4. **Linguagem Compartilhada (Storytelling):** O código técnico é mapeado para um vocabulário amigável para nutricionistas, gestores e a banca examinadora (conforme `docs/GLOSSARIO.md`).

---

## 📅 Linha do Tempo e Entradas do Diário

### [Entrada #001] – 01/11/2026: Consolidação do Novo Artefato e Camadas Bronze & Silver
* **Contexto e Motivação:**
  * Diagnóstico crítico nas iterações anteriores (`ARTEFATO` e `ARTEFATO - Claude`): detecção de vazamento temporal em defasagens móveis globais e necessidade de construir uma base definitiva, modular e sem dependências legadas.
  * Decisão de criar o repositório consolidado do zero em `ARTEFATO_ANALITICO_PREDITIVO`, com documentação de suporte à escrita da dissertação.
* **Ações Executadas:**
  1. **Governança Documental:**
     * Criação de `docs/GLOSSARIO.md`: padronização de termos para nutricionista e gestores.
     * Criação de `docs/STORYTELLING_DISSERTACAO.md`: roteiro narrativo amarrando os capítulos da dissertação às decisões de código.
  2. **Camada Bronze (Dados Brutos):**
     * Cópia integral e auditada de dados de calendário (2023-2026), consumo diário, reservas discentes/planejamento da nutricionista, cardápio, custos financeiros e histórico horário de previsões de tempo arquivadas (`previous_runs_horario.csv` e `single_runs_horario.csv`).
     * Anonimização ética de eventos institucionais no calendário para conformidade com a LGPD e publicação acadêmica aberta.
  3. **Camada Silver & Classificação Determinística de Atípicos:**
     * Implementação dos módulos modulares: `src/core/config.py`, `src/core/terminology.py`, `src/core/atipicos.py` e `src/core/bronze_silver.py`.
     * Mapeamento de 1.461 dias de calendário em blocos contínuos de aula com contagem de início e fim de período letivo.
     * Identificação causal de 68 candidatos a atípicos a partir do nível basal recente ($\bar{y}_{base}$):
       * **582 dias normais** ($\pm 40\%$ da referência).
       * **32 dias de quedas previstas por calendário** (vésperas de feriado, fim de semestre) $\rightarrow$ *mantidos no treino*.
       * **20 dias de pico de consumo** (cardápios preferidos) $\rightarrow$ *mantidos no treino*.
       * **634 dias elegíveis para treino de Machine Learning**.
       * **22 dias isolados do treino** (3 de greve, 8 fora do calendário letivo e 11 quedas severas sem explicação causal).
     * Geração de arquivos em formatos duais: Parquet (performance) e CSV (auditabilidade).
  4. **Notebook 01:**
     * Criação de `notebooks/01_ingestao_bronze_silver.ipynb` com narrativa e visualização gráfica da série histórica e dos outliers destacados.
  5. **Versionamento:**
     * Inicialização do repositório Git com commit inicial `feat(init): estrutura inicial do artefato, bronze e camada silver com storytelling`.
* **Decisões Técnicas Registradas:**
  * *Decisão:* Não excluir outliers da base Silver; em vez disso, criar a coluna booleana `usar_no_treino` na tabela `silver_dias_classificados`. Isso garante transparência total e reprodutibilidade, mantendo o histórico completo intacto.

---

### [Entrada #002] – 01/11/2026: Camada Gold e Prova Matemática de Zero Data Leakage
* **Contexto e Motivação:**
  * O principal ponto fraco detectado nas primeiras versões da pesquisa foi o vazamento de dados decorrente de médias móveis calculadas globalmente.
  * O objetivo deste marco foi implementar uma engenharia de atributos matematicamente causal, orientada pela **sexta-feira de corte (`origem`)**, e provar por meio de auditoria adversarial automatizada que nenhuma informação futura vaza para o passado.
* **Ações Executadas:**
  1. **Módulo de Atributos Gold (`src/core/gold_features.py`):**
     * Construção de atributos causais resumidos por data de emissão: nível basal recente (medianas de 10 e 20 dias, EWMA $\alpha=0.2$), dispersão recente, histórico do mesmo dia da semana e distância ordinal para início e fim do bloco letivo contínuo.
     * Suavização empírica Bayesiana para o efeito do cardápio (pratos, proteínas e métodos de preparo), evitando sobreajuste em receitas raras.
     * Separação estrita dos dois horizontes decisórios:
       * **Horizonte $T+1$ (Preparo Diário):** 705 linhas x 79 colunas (3 a 7 dias de antecedência, com cardápio e quantitativo humano da nutricionista/reservas discentes).
       * **Horizonte $T+2$ (Compras Quinzenais):** 705 linhas x 77 colunas (10 a 14 dias de antecedência, estritamente autônomo, sem permissão de utilizar planos humanos).
     * Geração do dicionário de metadados em `dados/gold/gold_dicionario_atributos.csv`.
  2. **Módulo de Auditoria Automatizada (`src/core/audit_leakage.py`):**
     * Bateria de 3 grandes suítes de testes:
       * *Estrutura Operacional:* Validação de emissões 100% em sextas-feiras, antecedências válidas e garantia de ausência de planos humanos em $T+2$ [APROVADO].
       * *Qualidade Numérica:* Sem infinitos, níveis basais estritamente positivos e ausência de atributos constantes [APROVADO].
       * *Teste Unitário Adversarial Anti-Vazamento:* Embaralhamento aleatório de consumos futuros em 5 sextas-feiras representativas de diferentes semestres (2023 a 2026). Comprovada a invariância matemática de todos os preditores (`assert_frame_equal`) [APROVADO].
  3. **Notebook 02:**
     * Criação de `notebooks/02_camada_gold_e_auditoria_vazamento.ipynb` documentando a falha da validação cruzada tradicional, o protocolo de sexta-feira e a execução interativa dos testes para a banca examinadora.
* **Decisões Técnicas Registradas:**
  * *Decisão:* Tratar variáveis climáticas observadas (`chuva_mm`, `temp_max`) como "Oráculo Ex-Post", utilizadas exclusivamente para análise retrospectiva de sensibilidade. Para o treinamento e inferência preditiva, os modelos utilizam unicamente as variáveis de previsão horária arquivadas no momento da emissão (`previsao_chuva_mm`, `previsao_temp_max`).

---

### [Entrada #003] – 01/11/2026: Análise Exploratória (EDA), Métricas Operacionais e Suíte de 12 Figuras
* **Contexto e Motivação:**
  * Investigação empírica dos fatores que explicam a demanda de refeições e comparação quantitativa entre o regime de Reservas Discentes (2023-2025) e o Planejamento Técnico da Nutricionista (2026).
  * Foco estrito em métricas operacionais de precisão (MAE, RMSE, WAPE) e balanço de segurança alimentar (sobra potencial vs. déficit potencial), omitindo o conceito de viés e postergando o custo financeiro para os capítulos finais de impacto na gestão pública.
  * Expansão da produção gráfica para compor a suíte completa de evidências dos resultados da dissertação.
* **Ações Executadas:**
  1. **Módulo de EDA (`src/core/eda.py`):**
     * Cálculo de métricas operacionais de acurácia e balanço de atendimento discente.
     * Ranking de correlações lineares (Pearson) e relativas para os 18 atributos preditores candidatos.
     * Geração da **Suíte de 12 Figuras Acadêmicas** em alta resolução (150 DPI) salvas em `resultados/figuras/eda/`:
       * *Figura 01:* Confronto dos Métodos Humanos (Dispersão x Consumo Real: Reservas Discentes vs Nutricionista).
       * *Figura 02:* Balanço Acumulado de Sobras vs Faltas Potenciais (Volume de refeições em sobra e falta).
       * *Figura 03:* Perfil Semanal de Consumo Facetado por Ano (Consistência longitudinal de 2023 a 2026).
       * *Figura 04:* Sazonalidade Mensal do Consumo ao Longo do Calendário Letivo.
       * *Figura 05:* Desmobilização Acadêmica no Término do Semestre (Queda na reta final).
       * *Figura 06:* Demanda e Frequência por Tipo de Proteína no Cardápio.
       * *Figura 07:* Consumo Conforme a Técnica Culinária de Preparo.
       * *Figura 08:* Sensibilidade Climática por Faixas de Precipitação Prevista.
       * *Figura 09:* Relação Térmica entre Temperatura Máxima e Consumo Real.
       * *Figura 10:* Matriz de Correlação Multivariada (Heatmap).
       * *Figura 11:* Autocorrelação Serial do Consumo Escolar (Lags 1 a 20 dias úteis).
       * *Figura 12:* Tornado Chart - Ranking Geral de Poder Preditivo dos Atributos.
  2. **Principais Descobertas das Métricas:**
     * *Acurácia Comparada:* O método da nutricionista (2026) supera amplamente as reservas discentes (2023-2025), com correlação saltando de $r=0.574$ para $r=0.772$, MAE caindo de $27.0$ para $23.7$ e WAPE de $19.9\%$ para $13.6\%$.
     * *Risco de Desabastecimento:* No regime discente, em $65.6\%$ dos dias houve déficit potencial (acumulando 9.332 refeições faltantes no agendamento). Com a nutricionista, a taxa de déficit potencial cai para $39.6\%$ (apenas 994 refeições acumuladas).
     * *Preditores Críticos:* Nível basal recente ($r=0.658$), inércia do mesmo dia da semana anterior ($r=0.656$) e dias restantes até o término do semestre ($r=0.334$) são os motores principais do consumo.
     * *Desmobilização de Fim de Semestre:* Nos últimos 3 dias letivos do semestre, o consumo médio desaba em **56.1%** (de 149.0 para 65.4 refeições), comprovando a importância da feature ordinal de calendário.
     * *Desmistificação do Clima:* Previsões de chuva ($r=+0.041$) e temperatura máxima ($r=-0.029$) possuem correlação linear insignificante com o consumo ordinário escolar, justificando testar o clima como cenário complementar, sem torná-lo dependência obrigatória.
  3. **Notebook 03:**
     * Atualização do notebook `notebooks/03_analise_exploratoria_eda.ipynb` integrando as 12 figuras e texto explicativo pronto para a dissertação.
* **Decisões Técnicas Registradas:**
  * *Decisão:* Remover métricas de viés e cálculos de custo em R$ das análises exploratórias, mantendo a EDA focada estritamente no comportamento físico da demanda e nas métricas de erro (MAE, RMSE, WAPE) e balanço operacional (sobra vs. déficit de refeições). Os custos em reais serão calculados no módulo final de impacto financeiro e tomada de decisão.

---

### [Entrada #004] – 01/11/2026: Torneio Walk-Forward com Taxonomia Completa de Ensembles e Auditoria de Modelos
* **Contexto e Motivação:**
  * Implementação da taxonomia completa de algoritmos de Machine Learning e métodos de Ensemble solicitada para a dissertação:
    * *Bagging:* Random Forest e ExtraTrees (amostragem bootstrap independente em paralelo).
    * *Boosting:* Gradient Boosting, AdaBoost, LightGBM, XGBoost e CatBoost (aprendizado sequencial com correção de resíduos).
    * *Voting:* Votação heterogênea por média ponderada (Ridge + ExtraTrees + CatBoost).
    * *Stacking:* Meta-regressão com estimadores de primeiro nível e meta-modelo regularizado não-negativo.
    * *Blending:* Divisão em treino base e holdout temporal recente sem vazamento com otimização NNLS.
    * *Cascading:* Encadeamento sequencial em dois estágios (âncora linear regularizada para tendência global + Gradient Boosting no resíduo não-linear).
  * Execução da validação *Walk-Forward* estrita ao longo de 2026 com janela móvel de 12 meses de treino em $T+1$ (Preparo, 3 a 7 dias) e $T+2$ (Compras quinzenais, 10 a 14 dias, 100% autônomo).
* **Ações Executadas:**
  1. **Auditoria de Não-Vazamento dos Modelos (`src/core/test_zero_leakage_models.py`):**
     * Execução automatizada e formalmente aprovada em 3 suítes:
       * *Isolamento de Alvos e Restrições de Horizonte:* Sem variáveis de consumo ou oráculos nas matrizes $X$, e $T+2$ estritamente desprovido de planos humanos [APROVADO].
       * *Fronteira Temporal em 100% das Semanas:* $\max(\text{treino}) < \text{origem}$ em todas as 320 semanas auditadas [APROVADO].
       * *Invariância Adversarial (Teste Cego com Futuro Corrompido):* Predições idênticas com $\Delta = 0.000000$ em emissões de 2024, 2025 e 2026 mesmo após corromper até 451 dias de dados futuros [APROVADO]. Relatório salvo em `resultados/auditoria/relatorio_teste_vazamento_modelos.json`.
  2. **Execução do Torneio Walk-Forward (`src/core/models.py`):**
     * Execução sequencial robusta sem concorrência de processos no Windows, gerando 17.731 predições detalhadas.
     * Exportação de `metricas_torneio_modelos.csv`, `metricas_torneio_modelos.parquet` e `predicoes_detalhadas_torneio.parquet` na pasta `resultados/modelos/`.
     * *Destaques do Torneio:*
       * Em $T+1$ (Cenário 4 - com nutricionista): o modelo **Lasso** alcançou MAE=21.66 e WAPE=12.38%, e o **Ensemble Voting Heterogêneo** atingiu MAE=21.85 e WAPE=12.48%, superando o plano humano da nutricionista (MAE=23.7).
       * Em $T+2$ (Compras 100% autônomas, sem estimativas humanas): o **Ensemble ExtraTrees** alcançou MAE=24.21 e WAPE=13.84%, e o **Ensemble Voting** alcançou MAE=24.13 e WAPE=13.79%, comprovando a viabilidade de automatização quinzenal de suprimentos.

---

### [Entrada #005] – 01/11/2026: Margens Conformais, Modelo Híbrido e Separação Retrospectiva (Fev-Ago) vs. Prospectiva (Set-Out)
* **Contexto e Motivação:**
  * Atendimento à diretriz metodológica de segregar os resultados empíricos de 2026 em dois blocos transparentes para a dissertação:
    * **Período Retrospectivo (Fevereiro a Agosto de 2026 - $N=121$ dias):** A coorte de referência avaliada na qualificação, cobrindo o ablation study de cenários, ranking de ensembles e calibração das políticas operacionais.
    * **Período Prospectivo (Setembro e Outubro de 2026 - $N=28$ dias):** Teste cego *out-of-time* em campo, avaliando o estresse climático de enchentes severas em setembro e a estabilização da operação regular em outubro.
* **Ações Executadas:**
  1. **Módulo de Margens e Políticas Operacionais (`src/core/safety_margins.py`):**
     * *Modelo Híbrido Adaptativo ($T+1$):* $\hat{y}_{\text{híbrido}} = 0.50 \cdot \hat{y}_{\text{IA}} + 0.50 \cdot \hat{y}_{\text{nutri}}$, alcançando MAE=21.43 no retrospectivo e MAE=8.36 em outubro regular.
     * *Margens Conformais de Segurança:* Adição de buffers empíricos (+10, +12, +15, +19 refeições). A política Conformal P85 (+12 refeições) derrubou a frequência de dias com déficit de 43.4% para 15.7%, garantindo a segurança alimentar com acréscimo controlado de sobras.
     * *Custo Financeiro Potencial de Desperdício:* Precificação a R$ 14,00 por refeição excedente, permitindo o mapeamento da curva de trade-off entre risco de falta e custo fiscal.
  2. **Geração das Tabelas Oficiais para a Dissertação (`resultados/tabelas_dissertacao/`):**
     * `tabela_03_desempenho_cenarios_retrospectivo.csv`: Ablation study completo nos 4 cenários e 2 horizontes.
     * `tabela_04_modelos_vs_nutricionista_t1_retrospectivo.csv`: Confronto com o especialista humano.
     * `tabela_05_politicas_margens_retrospectivo.csv`: Avaliação de políticas com margens conformais e custos em R$.
     * `tabela_06_modelos_t2_compras_retrospectivo.csv`: Desempenho autônomo para compras quinzenais.
     * `tabela_07_desempenho_prospectivo_set_out.csv`: Bimestre prospectivo completo.
     * `tabela_08_resiliencia_climatica_setembro_enchentes.csv`: Resiliência em dias regulares vs. dias de crise de enchente.
     * `tabela_10_sintese_comparativa_retrospectivo_vs_prospectivo.csv`: Quadro síntese confrontando os regimes.
  3. **Suíte Completa de 11 Figuras para a Dissertação (`resultados/figuras/dissertacao/`):**
     * *Figuras 18 a 23 (Retrospectivo):* Ablation study, confronto da taxonomia de ensembles, dispersão IA vs Nutricionista, curva de equilíbrio do híbrido, margens conformais e curva de trade-off custo vs falta.
     * *Figuras 24 a 28 (Prospectivo e Síntese):* Série temporal com precipitação e catraca real, resiliência nas enchentes de setembro, estabilidade no retorno de outubro, balanço acumulado de sobras/faltas e síntese comparativa de MAE.
  4. **Notebooks 04 e 05:**
     * Conclusão de `notebooks/04_torneio_modelos_walk_forward.ipynb` e `notebooks/05_modelo_hibrido_e_margens_seguranca.ipynb` com narrativa completa e exibição de figuras.

---

### [Entrada #006] – 01/11/2026: Aplicação Web Streamlit, Testes Automatizados e Conclusão do Artefato DSR
* **Contexto e Motivação:**
  * Entrega do software operacional interativo exigido pela metodologia *Design Science Research* (DSR) para a nutricionista e os gestores da instituição pública.
  * Validação formal automatizada via `pytest` cobrindo integridade de dados, artefatos acadêmicos, sintaxe do app e a auditoria matemática adversarial de *Zero Data Leakage*.
* **Ações Executadas:**
  1. **Aplicação Web Interativa (`src/app/main.py` e `executar_app.py`):**
     * Interface moderna com design responsivo (Pills de navegação, Altair charts, CSS institucional verde `#008744`).
     * **Aba 1 (Visão Executiva & Histórico):** KPIs dinâmicos, filtros de dias letivos/semana, série temporal interativa e galeria das 12 figuras da EDA.
     * **Aba 2 (Preparo Diário T+1):** Desempenho retrospectivo do torneio, simulador interativo de margens conformais (slider de buffer de segurança) e custo financeiro a R$ 14,00/ref.
     * **Aba 3 (Compras Quinzenais T+2):** Projeção autônoma sem plano humano para pedidos quinzenais com antecedência licitatória.
     * **Aba 4 (Auditoria de Não-Vazamento):** Painel de integridade com botão para reexecutar testes adversariais ao vivo para a banca examinadora.
     * **Aba 5 (Resiliência Climática & Teste Cego):** Análise do modelo congelado em 31/08/2026 e resiliência frente às inundações de setembro.
  2. **Suíte de Testes Automatizados (`pytest`):**
     * Criação de `tests/test_zero_leakage.py`, `tests/test_data_integrity.py` e `tests/test_app_syntax.py`.
     * 100% de aprovação (4/4 testes aprovados com sucesso).
* **Decisões Técnicas Registradas:**
  * O artefato está plenamente operacional, auditado, reprodutível e pronto tanto para uso em campo quanto para defesa na banca examinadora.

---

### [Entrada #007] – 01/11/2026: Correção da Avaliação, Modelo Congelado e Auditoria Reescrita
* **Contexto e Motivação:**
  * Revisão independente confirmou que os atributos e o treino respeitam a sexta de corte, mas encontrou problemas na medição do erro: o filtro `usar_no_treino` retirava dias do teste usando o consumo do próprio dia, valores do período Fev-Ago vinham do torneio de Fev a Out, previsões e margens estavam digitadas no código, e os testes de auditoria não conseguiam reprovar.
* **Premissas Declaradas pelo Autor:**
  * Fev-Ago/2026 é retrospectivo. Set-Out/2026 é prospectivo, com **ExtraTrees** congelado em 31/08/2026.
  * A emissão ocorre na sexta à tarde. Cardápio, calendário e previsão de clima estão disponíveis nos dois horizontes, e o quantitativo institucional existe apenas no $T+1$: reservas discentes até 2025 e estimativa da nutricionista em 2026, ambas fechadas até a sexta à tarde.
  * O registro de dia atípico é feito depois do consumo e serve para que o dia não contamine as previsões seguintes.
* **Ações Executadas:**
  1. **Avaliação:** `usar_no_treino` permanece no treino e saiu do teste. O resultado principal usa todos os dias (123 em Fev-Ago e 33 em Set-Out), e o recorte sem evento registrado aparece ao lado (28 dias em Set-Out).
  2. **Período Prospectivo:** o modelo congelado passou a ser o resultado principal, e o walk-forward virou análise de sensibilidade. Registro de reprocessamento: as métricas foram geradas em 01/11/2026; o modelo é reconstruído com dados até 31/08, e sua especificação difere do ExtraTrees do torneio.
  3. **Valores Digitados:** tabelas, títulos e figuras agora são calculados a partir das previsões. A afirmação de 30,56 refeições foi substituída por 14,60.
  4. **Margens:** o P85 fixo de +12 foi substituído por quantis empíricos de Fev-Ago (P85 do híbrido: +21 refeições), aplicados sem recalibrar em Set-Out. O termo "conformal" foi retirado.
  5. **Comparação Estatística:** nova tabela com bootstrap pareado por semana, 10.000 reamostragens.
  6. **Cenário 3:** nova tabela com o modelo congelado nos 33 dias e nos dias de enchente.
  7. **Auditoria:** testes reescritos sobre a avaliação real, com vazamentos injetados para provar que reprovam.
* **Decisões Técnicas Registradas:**
  * *Decisão:* Não mudar o corte para `data <= origem`. O teste de sensibilidade não mostrou ganho claro.
  * *Decisão:* Não alterar a proteção dos lags nesta versão. Retirar também o dia 11/09/2026, registrado como atípico mas acima do limiar de 60%, muda atributos de 21 linhas de Set-Out e fica como proposta para uma nova versão.
  * *Decisão:* Não trocar o ExtraTrees por modelos com resultado melhor em Set-Out, para não escolher depois de ver o período prospectivo.
* **Pendências:**

---

### [Entrada #008] – 02/11/2026: Unificação do Aplicativo aos Modelos Auditados
* **Contexto e Motivação:**
  * O aplicativo treinava modelos próprios em uma base antiga, com clima observado, e exibia métricas fixas. O simulador usava modelos treinados até 31/08 mesmo para datas anteriores.
  * Testes isolados com 80 configurações de atributos não justificaram mudança: a configuração escolhida em Fev-Ago não se sustentou em Set-Out, e retirar o clima ou ajustar a escala do quantitativo não trouxe ganho. A escolha foi gravada antes de calcular Set-Out.
* **Ações Executadas:**
  1. **Modelos Servidos:** Preparo ($T+1$): Híbrido 50/50 com ExtraTrees Cenário 2. Compras ($T+2$): ExtraTrees Cenário 2, sem quantitativo. Ambos congelados em 31/08/2026.
  2. **Atributos Operacionais:** O app calcula a emissão com as mesmas funções da camada Gold. Dias com registro de atípico ficam fora do histórico das próximas previsões.
  3. **Simulador:** Corta os dados na sexta da emissão. Datas até 31/08/2026 usam walk-forward e as demais usam o modelo congelado. O clima é sempre o previsto.
  4. **Quantitativo:** Removidos o uso no $T+2$ e o valor fixo de 182,7. Sem quantitativo, o resultado usa apenas o ExtraTrees.
  5. **Margens e Textos:** Margens lidas das tabelas calculadas, e textos antigos substituídos por valores calculados.
  6. **Senhas:** Retiradas do código. A edição exige `st.secrets`.
* **Limitação Registrada:**
  * Nos Cenários 3 e 4 de datas futuras, o app recebe só a chuva e a temperatura diárias. Chuva da manhã e acumulado de 3 dias são imputados.
  * A base `gold_dataset_ml_limpo` deixou de ser usada pelo código.

---

### [Entrada #009] – 02/11/2026: Fechamento dos Notebooks e das Figuras dos Métodos Humanos
* **Ações Executadas:**
  1. **Figuras 01 e 02:** passaram a medir o erro dos métodos humanos em todos os dias com consumo (442 dias em 2023-2025 e 156 em 2026). Os valores substituem os da Entrada #003: nutricionista com $r=0.619$, MAE 29,0 e WAPE 17,1%; reservas com $r=0.575$, MAE 27,2 e WAPE 20,2%.
  2. **Notebooks:** o 04 reúne a avaliação retrospectiva de Fev-Ago, com torneio, híbrido e margens. O 05 reúne a avaliação prospectiva de Set-Out. Treino e teste seguem no mesmo fluxo, com a regra de treino inalterada.
  3. **Saídas:** os cinco notebooks passaram a ser salvos com as tabelas e figuras.
* **Decisões Técnicas Registradas:**
  * *Decisão:* As Figuras 03 a 12 continuam descrevendo dias regulares, o que está declarado no notebook 03.
