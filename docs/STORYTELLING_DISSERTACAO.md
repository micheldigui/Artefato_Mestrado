# Roteiro de Storytelling para a Dissertação de Mestrado

**Título de Trabalho:** Artefato Analítico-Preditivo para Planejamento de Demanda e Mitigação do Desperdício em Restaurante Institucional de Ensino  
**Pesquisador:** Michel Silva de Souza  
**Programa:** Mestrado Profissional em Educação Profissional e Tecnológica (ProfEPT) / Gestão Pública  

---

## 1. Visão Geral da Narrativa (A Jornada da Pesquisa)

A dissertação conta uma história clássica de **pesquisa aplicada baseada em Design Science Research (DSR)**: a união entre rigor científico (Ciência de Dados, Machine Learning sem vazamento temporal) e utilidade prática para a administração pública escolar (gestão de estoques, combate ao desperdício e garantia da segurança alimentar dos estudantes).

```
[Problema Operacional] 
       │
       ▼
[Diagnóstico de Dados & Lições Aprendidas] (Auditoria de vazamento e realidade do campus)
       │
       ▼
[Engenharia de Recursos Temporalmente Estrita] (Corte operacional às sextas-feiras)
       │
       ▼
[Torneio de Modelos Walk-Forward] (Baseline vs ML vs Modelo Híbrido)
       │
       ▼
[Políticas Conformal de Proteção Alimentar] (Margens de segurança anti-fome)
       │
       ▼
[O Artefato Computacional] (Aplicação interativa Streamlit com simulação)
```

---

## 2. Estrutura por Capítulos / Seções da Dissertação

### Capítulo 1: O Problema e o Contexto Institucional
* **O Dilema da Nutricionista:**
  * O Instituto Federal oferece alimentação gratuita ou subsidiada para estudantes em vulnerabilidade social.
  * O dilema diário: **errar para mais** gera desperdício de recursos públicos escassos (refeições descartadas e compras excessivas); **errar para menos** gera déficit alimentar (estudantes vulneráveis sem almoço).
* **As Duas Decisões Descasadas no Tempo:**
  1. *Decisão de Compras ($T+2$ - quinzenal):* Antecedência de 10 a 14 dias para licitação/pedidos de hortifrúti e proteínas. Não há cardápio detalhado fixado nem estimativa humana disponível.
  2. *Decisão de Preparo ($T+1$ - semanal):* Antecedência de 3 a 7 dias, decidida na sexta-feira para os pratos da próxima semana.
* **Objetivo do Artefato:** Construir um sistema inteligente que atue como assistente decisório para a nutricionista e a gestão do restaurante.

---

### Capítulo 2: Fundamentação e Rigor Metodológico (Zero Data Leakage)
* **A Armadilha do Vazamento Temporal em Séries Temporais:**
  * Por que a validação cruzada tradicional ($K$-Fold aleatória) é inaceitável em dados temporais.
  * O erro sutil descoberto nas primeiras versões: calcular médias móveis ou defasagens autorregressivas no dataset completo antes de separar o treino, o que "vaza" o consumo de segunda-feira para a previsão de terça-feira da mesma semana planejada.
* **O Protocolo Operacional Estrito (Corte às Sextas-feiras):**
  * Toda previsão da semana seguinte ($T+1$) e da semana subsequente ($T+2$) é emitida rigorosamente na **sexta-feira anterior**.
  * **O Teste Adversarial de Não-Vazamento:** Implementação de teste unitário automatizado que embaralha os consumos futuros e comprova que nenhum atributo ou decisão preditiva passada é alterada.

---

### Capítulo 3: Da Camada Bronze à Silver (Qualidade dos Dados e Outliers)
* **Arquitetura Medalhão:**
  * **Bronze:** Ingestão de dados brutos (calendários acadêmicos, planilhas de consumo diário, reservas discentes/planejamento da nutricionista, cardápios e dados meteorológicos horários).
  * **Silver:** Limpeza, anonimização de eventos sensíveis (LGPD e ética na pesquisa), tipagem estrita e integração na `silver_fato_diaria`.
* **Classificação Determinística de Dias Atípicos:**
  * Como separar o joio do trigo: não eliminar dados arbitrariamente.
  * Criação de regras reproduzíveis: dias de greve, eventos de calendário previstos, fins de blocos letivos e dias de força maior (alagamentos).
  * Definição clara do subconjunto de treinamento que ensina os algoritmos.

---

### Capítulo 4: Camada Gold e Engenharia de Atributos
* **Engenharia de Atributos Baseada no Domínio:**
  * *Nível Basal Recente:* Medianas de 10 e 20 dias úteis e médias móveis exponenciais ponderadas.
  * *Sazonalidade Acadêmica:* Dias desde o início e dias até o fim do bloco letivo contínuo (capturando a curva de evasão/desistência temporária de fim de semestre).
  * *Efeito do Cardápio via Suavização Bayesiana:* Em vez de one-hot encoding ingênuo, cálculo de estimativas empíricas de Bayes (Empirical Bayes Prior) para pratos e proteínas, evitando overfitting em pratos raros.
  * *O Papel Real do Clima:* Uso de previsões meteorológicas horárias arquivadas no momento da previsão.

---

### Capítulo 5: Torneio de Algoritmos e Validação Walk-Forward
* **Avaliação em Janela Deslizante / Expansiva:**
  * Treinamento com histórico de 12 meses anteriores à emissão.
  * Avaliação retrospectiva em 2024-2025 e prospectiva em 2026.
* **Resultados dos Cenários (Ablation Study):**
  * *Cenário 1 (Calendário + Consumo)* $\rightarrow$ Linha de base autônoma estável.
  * *Cenário 2 (+ Cardápio)* $\rightarrow$ Ganho expressivo em dias com pratos de alta aceitação.
  * *Cenário 3 (+ Clima)* $\rightarrow$ Redução de erro marginal (desmistificando o mito de que o clima é o fator principal na rotina escolar).
  * *Cenário 4 (+ Nutricionista)* $\rightarrow$ O salto de acurácia com a presença do especialista humano.
* **O Modelo Híbrido Adaptativo:**
  * O algoritmo campeão não compete com a nutricionista: ele **combina** a sensibilidade empírica dela com a memória matemática das árvores de decisão.

---

### Capítulo 6: Margens de Segurança Conformal e Gestão Pública
* **Transformando Estatística em Segurança Alimentar:**
  * A previsão pontual (média) garante que em 50% dos dias haverá déficit se a demanda flutuar para cima.
  * Introdução de **Margens de Proteção Alimentar Calibradas** (+12 a +18 refeições), garantindo atendimento discente com 90% a 95% de nível de serviço.
* **Análise de Desperdício e Economia Financeira:**
  * Quantificação do desperdício evitado em quilogramas de alimento e em reais (R$).

---

### Capítulo 7: O Artefato Tecnológico (Streamlit) e Demonstração
* **Apresentação da Interface:**
  * Painel de Controle Executivo para a gestão do campus.
  * Simulador de Cenários Walk-Forward para a banca e operadores testarem qualquer data histórica.
  * Módulo de Compras Quinzenais autônomo.
  * Auditoria de Dados e Teste de Vazamento interativo.

