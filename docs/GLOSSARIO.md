# Glossário do Artefato Analítico-Preditivo

Este glossário estabelece a linguagem padronizada adotada em todas as camadas do projeto: tabelas, métricas, figuras e telas da aplicação. O módulo `src/core/terminology.py` centraliza as conversões de nomes técnicos para a linguagem compreensível pela nutricionista, pela gestora e pela banca examinadora.

---

## 1. Os Dois Horizontes Decisórios Operacionais

| Termo Operacional (Interface / Dissertação) | O que significa na rotina da escola pública | Termo Técnico (Código) |
| :--- | :--- | :--- |
| **Preparo Diário (Semana Seguinte)** | Na sexta-feira, prever a quantidade de almoços a serem servidos de segunda a sexta da semana que vem (3 a 7 dias de antecedência). | $T+1$ (curto prazo) |
| **Compras de Suprimentos (Quinzenal)** | Na sexta-feira, prever o consumo da semana posterior à próxima para guiar pedidos de carnes, hortifrúti e insumos perecíveis (10 a 14 dias de antecedência). | $T+2$ (médio prazo) |
| **Sexta-feira de Corte / Planejamento** | O marco temporal estrito de corte. Nenhuma informação posterior a esta sexta-feira pode ser utilizada para prever as semanas seguintes (Garantia de Não-Vazamento Temporal). | `origem`, emissão temporal |

---

## 2. Cenários de Informação (Ablation Study)

O artefato implementa uma abordagem em camadas de informação, permitindo operar mesmo quando fontes externas (como cardápio ou previsão do tempo) não estiverem disponíveis:

| Cenário | O que utiliza | Condição de Aplicação |
| :--- | :--- | :--- |
| **Cenário 1: Histórico e Calendário Escolar** | Consumos passados, dia da semana, mês, posição no bloco de aulas e eventos acadêmicos. | Sempre disponível (Linha de base autônoma). |
| **Cenário 2: Inclusão do Cardápio** | Cenário 1 + prato principal, tipo de proteína, método de preparo e opções vegetarianas. | Quando o cardápio semanal for publicado pela nutricionista. |
| **Cenário 3: Inclusão da Previsão Meteorológica** | Cenário 2 + previsão de chuva (volume e período matutino) e temperatura máxima prevista. | Aplicável primordialmente no horizonte de Preparo ($T+1$). |
| **Cenário 4: Integração com o Planejamento Humano** | Cenário anterior + quantitativo estimado pela nutricionista (2026) ou reservas discentes (2023-2025). | Disponível exclusivamente em $T+1$ (quando há estimativa operacional cadastrada). |

---

## 3. Demanda e Metodologia de Previsão

| Termo Operacional | Definição no Contexto da Pesquisa | Termo Matemático / Técnico |
| :--- | :--- | :--- |
| **Consumo Real (Refeições Servidas)** | Total efetivo de refeições distribuídas na catraca do refeitório. | Alvo de regressão ($y$) |
| **Consumo de Referência (Nível Basal)** | Demanda estimada antes da data usando a mediana móvel de dias normais e o mesmo dia da semana recente. | Nível basal / Baseline ($\bar{y}_{base}$) |
| **Consumo Relativo à Referência** | Variação percentual em relação ao nível basal. Permite ao modelo aprender padrões invariantes a oscilações no número de matrículas. | Razão normalizada ($y / \bar{y}_{base}$) |
| **Modelo Híbrido Adaptativo** | Combinação ponderada entre o algoritmo de aprendizado de máquina e a estimativa empírica da nutricionista, explorando o melhor de ambos. | Ensemble ponderado ($w_1 \cdot \hat{y}_{ML} + w_2 \cdot \hat{y}_{nutri}$) |
| **Margem de Proteção Alimentar (Conformal)** | Acréscimo estatisticamente calibrado (+12 a +18 refeições) para mitigar o risco social de falta de comida para os estudantes vulneráveis. | Margem residual empírica ($q_{1-\alpha}$) |
| **Bloco Contínuo de Aulas** | Período ininterrupto de aulas entre recessos, férias ou greves. Crucial para capturar quedas de frequência de fim de semestre. | Regime letivo / Bloco acadêmico |

---

## 4. Classificação dos Dias e Auditoria de Outliers

Para não contaminar o aprendizado dos modelos com eventos anômalos que não se repetirão, os dias são classificados de forma determinística e transparente:

| Classificação Operacional | Definição Prática | Tratamento no Treinamento |
| :--- | :--- | :--- |
| **Dia Típico / Normal** | Demanda dentro da faixa esperada ($\pm 40\%$ da referência). | Treina todos os modelos |
| **Queda Prevista por Calendário** | Redução previsível (ex: semanas de provas, vésperas de feriado, início ou encerramento de bloco letivo). | Treina o algoritmo (é previsível) |
| **Pico Previsto de Demanda** | Aumento esperado devido a cardápio altamente preferido (ex: feijoada, strogonoff) ou evento acadêmico especial. | Treina o algoritmo |
| **Dia Atípico Imprevisível** | Evento de força maior (ex: tempestade severa com alagamento, pane de transporte, queda drástica < 30 ref). | Preservado na base histórica, mas excluído do treino |
| **Período de Greve** | Paralisação com funcionamento restrito ou atendimento irregular de alunos. | Preservado para auditoria, excluído do treino preditivo |
| **Refeições Fora do Calendário Letivo** | Dias não letivos oficiais que tiveram fornecimento esporádico de refeição. | Não integra a modelagem de rotina |

---

## 5. Métricas de Gestão Pública e Desperdício

| Métrica | Significado no Restaurante Institucional | Fórmula / Conceito |
| :--- | :--- | :--- |
| **MAE (Erro Médio Absoluto)** | Erro médio em número de refeições por dia. | $\frac{1}{n}\sum |y - \hat{y}|$ |
| **WAPE (Erro Percentual Ponderado)** | Percentual de erro ponderado pelo volume total (não explode em dias de baixo consumo como o MAPE). | $\frac{\sum |y - \hat{y}|}{\sum y} \times 100\%$ |
| **Sobra Potencial (Desperdício)** | Excesso de comida preparada além do consumo real. Representa custo financeiro e descarte. | $\max(0, \hat{y} - y)$ |
| **Déficit Potencial (Risco Social)** | Falta de comida preparada para atender os alunos presentes. Representa risco de fome para estudantes em vulnerabilidade. | $\max(0, y - \hat{y})$ |
| **Custo do Desperdício Evitado** | Economia em reais calculada sobre o valor oficial do contrato por refeição terceirizada. | $\Delta \text{Sobra} \times \text{Valor Contratual}$ |

