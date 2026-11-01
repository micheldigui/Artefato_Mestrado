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

### [Planejamento Próximos Passos]
* **Marco 3:** Análise Exploratória de Dados (EDA) e isolamento dos fatores de influência (cardápio, dia da semana, calendário e clima).
* **Marco 4:** Torneio de Algoritmos em Validação Walk-Forward (12 meses de treino) comparando Baselines, Modelos Lineares, Árvores e Boosting.
* **Marco 5:** Modelo Híbrido Adaptativo (IA + Nutricionista) e Calibração de Margens de Segurança Conformal (+12 a +18 refeições).
* **Marco 6:** Desenvolvimento da Aplicação Streamlit Modular (`src/app/`) com Simulador Interativo, Visão Executiva e Módulo de Compras.
* **Marco 7:** Validação Prospectiva em 2026, exportação de tabelas/figuras finais e fechamento das evidências empíricas da dissertação.

