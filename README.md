# 🍲 Artefato Analítico-Preditivo para Planejamento de Refeições Estudantis

> **Pesquisa de Mestrado:** Artefato Analítico-Preditivo para Planejamento da Demanda e Mitigação do Desperdício em Restaurante Institucional de Ensino  
> **Autor / Pesquisador:** Michel Silva de Souza ([@micheldigui](https://github.com/micheldigui))  
> **Metodologia:** *Design Science Research* (DSR)  
> **Campo de Aplicação:** Restaurante Estudantil de Instituição Pública de Ensino Profissional e Tecnológico  

---

## 📌 Visão Geral do Projeto

Este repositório contém o artefato computacional e analítico completo desenvolvido para solucionar o descompasso entre a quantidade de refeições planejadas e o consumo real em restaurantes universitários e escolares. 

O artefato foi concebido sob o rigor da **Design Science Research (DSR)**, respondendo a dois problemas operacionais concretos da escola pública:
1. **Horizonte $T+1$ (Preparo Diário):** Decisão na sexta-feira de corte com 3 a 7 dias de antecedência para a produção culinária, combinando previsão algorítmica, cardápio e estimativa técnica da nutricionista em um **Modelo Híbrido Calibrado** com **Margens Conformais de Segurança**.
2. **Horizonte $T+2$ (Compras de Insumos):** Decisão autônoma com 10 a 14 dias de antecedência para aquisição de gêneros alimentícios e carnes, **100% independente de planos humanos**.

---

## 🛡️ Rigor Científico e Zero Data Leakage

Uma das principais contribuições da pesquisa é a eliminação matemática de **qualquer vazamento temporal de dados** (*Zero Data Leakage*):
- **Corte Temporal Estrito:** A cada sexta-feira de emissão de corte ($\text{origem}$), é garantido que $\max(\text{treino}) < \text{origem}$.
- **Auditoria Adversarial Automatizada:** Testes unitários com contaminação proposital do consumo futuro comprovam invariância total ($\Delta = 0.000000$) nas previsões históricas.
- **Isolamento de Variáveis Ex-Post:** Variáveis climáticas observadas na data do almoço atuam apenas como "Oráculo Ex-Post" em estudos retrospectivos de ablação; para inferência preditiva, utilizam-se unicamente as previsões horárias arquivadas no momento da emissão.

---

## 🏆 Principais Descobertas e Resultados

1. **Torneio de Modelos e Ensembles:**
   - Implementação da taxonomia completa de aprendizado combinado: **Bagging** (*ExtraTrees, Random Forest*), **Boosting** (*Gradient Boosting, XGBoost, CatBoost, LightGBM, AdaBoost*), **Voting**, **Stacking**, **Blending** e **Cascading**.
2. **Modelo Híbrido Ótimo ($T+1$):**
   - A combinação convexa $\hat{y} = 0.50 \cdot \hat{y}_{\text{IA}} + 0.50 \cdot \hat{y}_{\text{nutri}}$ reduziu o erro da nutricionista de **24,86 para 21,43 refeições** no período retrospectivo e para **8,36 refeições** (WAPE de 4,08%) na operação estável de outubro.
3. **Margens Conformais de Segurança (Conformal Prediction):**
   - A adição de uma margem empírica de **+12 refeições (P85)** derrubou os dias com déficit de **43,4% para apenas 15,7%**, mantendo o custo fiscal de sobras em patamar ótimo de Pareto.
4. **Teste Cego Prospectivo com Modelo Campeão Congelado em 31/08/2026:**
   - O modelo campeão congelado na data de corte da qualificação superou o especialista humano isolado em todo o bimestre setembro-outubro (**MAE 18,29 vs 22,14 ref**).
5. **Resiliência Climática nas Enchentes de Setembro de 2026:**
   - Durante as cheias extremas no Alto Tietê, enquanto o humano insistiu em metas elevadas e gerou 683 refeições desperdiçadas em uma semana, a IA reduziu as estimativas e errou **30,56 refeições a menos que o humano**.

---

## 🚀 Como Executar a Aplicação Web Streamlit

A interface gráfica de apoio à decisão foi desenvolvida em Streamlit e pode ser executada localmente ou publicada em nuvem:

### 1. Clonar o Repositório e Instalar Dependências
```bash
git clone https://github.com/micheldigui/Artefato_Mestrado.git
cd Artefato_Mestrado
pip install -r requirements.txt
```

### 2. Iniciar a Aplicação
```bash
python executar_app.py
```
*(ou diretamente via `streamlit run app.py`)*

Acesse a interface no navegador em `http://localhost:8501`.

---

## 🧪 Suíte de Testes Automatizados

O repositório possui uma bateria formal de testes automatizados com `pytest` para validação contínua:
```bash
pytest -v
```

Testes inclusos:
- `test_zero_leakage.py`: Executa a auditoria adversarial e valida corte temporal em 100% das semanas.
- `test_data_integrity.py`: Valida a integridade das camadas Gold e a presença das tabelas e figuras oficiais.
- `test_app_syntax.py`: Garante que a aplicação web e seus módulos compilam sem erros.

---

## 📁 Estrutura do Repositório

```text
├── app.py                      <- Ponto de entrada raiz para Streamlit Cloud
├── executar_app.py             <- Iniciador da aplicação web
├── requirements.txt            <- Dependências do projeto
├── pytest.ini                  <- Configuração dos testes unitários
├── dados/                      <- Dados em camadas estruturadas
│   ├── bronze/                 <- Dados brutos originais e anonimizados
│   ├── silver/                 <- Dados tratados e dias atípicos classificados
│   └── gold/                   <- Matrizes preditivas causais T+1 e T+2
├── docs/                       <- Documentação metodológica
│   ├── DIARIO_DE_BORDO.md      <- Registro cronológico de todas as decisões
│   ├── GLOSSARIO.md            <- Vocabulário compartilhado e governança
│   └── STORYTELLING_DISSERTACAO.md <- Roteiro narrativo da dissertação
├── notebooks/                  <- Cadernos Jupyter reproduzíveis (01 a 05)
├── resultados/                 <- Entregáveis da dissertação
│   ├── auditoria/              <- Relatório JSON de não-vazamento
│   ├── figuras/                <- Suíte de 12 figuras EDA + 11 figuras Dissertação
│   ├── modelos/                <- Parquet e CSV do torneio walk-forward
│   └── tabelas_dissertacao/    <- 8 tabelas CSV estruturadas (Tabelas 03 a 10)
├── src/                        <- Código-fonte modular
│   ├── app/                    <- Aplicação web interativa (Streamlit)
│   └── core/                   <- Pipelines de dados, modelos e auditoria
└── tests/                      <- Suíte de testes automatizados
```

---

## 📄 Licença e Uso Acadêmico

Desenvolvido para fins de pesquisa acadêmica e modernização da gestão pública em restaurantes estudantis.  
Autor: **Michel Silva de Souza** ([micheldigui](https://github.com/micheldigui)).
