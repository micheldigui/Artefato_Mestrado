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

## 🛡️ Rigor Científico e Controle de Vazamento

A pesquisa separa o que estava disponível na sexta-feira de corte do que só seria conhecido depois:
- **Corte Temporal:** A cada sexta de emissão ($\text{origem}$), o treino usa apenas dias anteriores ($\max(\text{treino}) < \text{origem}$). Os atributos usam consumo anterior à sexta.
- **Informação Disponível na Emissão:** Cardápio, calendário, previsão de clima e, no $T+1$, o quantitativo institucional, fechado até a sexta à tarde (reservas discentes até 2025 e estimativa da nutricionista em 2026). O $T+2$ não usa quantitativo institucional.
- **Dias Atípicos:** O registro só existe depois do consumo. Dias atípicos ficam fora do treino, dias de consumo muito baixo ficam fora dos lags das previsões seguintes, e todos continuam visíveis na avaliação.
- **Auditoria Automatizada:** Os testes corrompem consumo, alvo e rótulos futuros, registram as chamadas reais de treino e confirmam que todos os dias com consumo entram no teste. Cada verificação é testada com vazamentos injetados de propósito.
- **Alcance:** A auditoria cobre a avaliação da pesquisa. A disponibilidade de reservas, plano e cardápio na sexta é declarada pelo autor.
- **Aplicativo:** Usa os mesmos modelos da pesquisa: Híbrido 50/50 no $T+1$ e ExtraTrees Cenário 2 no $T+2$, ambos congelados em 31/08/2026. O simulador corta os dados na sexta da emissão e nunca usa clima observado.

---

## 🏆 Principais Descobertas e Resultados

Os resultados principais usam todos os dias com consumo. O recorte sem evento registrado retira apenas os 5 dias de enchente de setembro, e foi definido depois do fato. Diferenças em que o intervalo de 95% do bootstrap por semana inclui zero não são tratadas como vitória.

1. **Torneio de Modelos e Ensembles:**
   - Implementação da taxonomia completa de aprendizado combinado: **Bagging** (*ExtraTrees, Random Forest*), **Boosting** (*Gradient Boosting, XGBoost, CatBoost, LightGBM, AdaBoost*), **Voting**, **Stacking**, **Blending** e **Cascading**.
2. **Período Retrospectivo (Fev-Ago/2026, 123 dias, walk-forward):**
   - Nutricionista com MAE 26,21, ExtraTrees 27,79 e Modelo Híbrido 50/50 23,78. A diferença entre o híbrido e a nutricionista não é estatisticamente clara (-2,43; intervalo de -5,60 a 0,57).
3. **Período Prospectivo (Set-Out/2026, modelo ExtraTrees congelado em 31/08/2026):**
   - Com todos os 33 dias: Nutricionista 39,48, ExtraTrees 34,00 e Híbrido 34,36.
   - Sem evento registrado (28 dias): Nutricionista 22,14, ExtraTrees 18,29 e Híbrido 17,36.
   - O híbrido supera a nutricionista nos dois recortes (-5,12; intervalo de -7,92 a -2,23 com 33 dias). Híbrido e ExtraTrees não têm diferença clara entre si.
   - As métricas foram reprocessadas em 01/11/2026. O modelo é reconstruído com dados até 31/08, e sua especificação difere do ExtraTrees do torneio.
4. **Margens Empíricas por Quantil:**
   - Calibrada em Fev-Ago, a margem P85 do híbrido é de **+21 refeições**. Ela reduz os dias com déficit de 46,3% para 13,8% em Fev-Ago, ao custo de mais sobras (R$ 53.200,00 contra R$ 27.692,00). Em Set-Out, sem recalibrar, a cobertura é de 84,8%.
5. **Enchentes de Setembro de 2026:**
   - Nenhum método antecipou os eventos. A nutricionista gerou 683 refeições de sobra nos 5 dias, e o ExtraTrees congelado, 610: uma diferença de **14,60 refeições por dia**, com previsões ainda muito acima do consumo real.
   - A previsão de clima sinalizou chuva forte em 09 e 10/09, mas não em 15 e 23/09, os dias de menor consumo. O Cenário 3 reduz pouco o erro nesses dias (117,60 contra 122,00) e piora os demais (19,54 contra 18,29).

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

### 3. Configurar as Senhas de Edição
O lançamento de dados exige senhas em `.streamlit/secrets.toml`, arquivo que não vai para o repositório, ou nos Secrets do Streamlit Cloud:
```toml
SENHA_NUTRI = "defina-uma-senha"
SENHA_ADMIN = "defina-outra-senha"
```
Sem essas chaves, o aplicativo abre em modo de leitura.

---

## 🧪 Suíte de Testes Automatizados

O repositório possui uma bateria formal de testes automatizados com `pytest` para validação contínua:
```bash
pytest -v
```

Testes inclusos:
- `test_zero_leakage.py`: Executa a auditoria da avaliação e confirma que vazamentos injetados são detectados.
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
│   └── tabelas_dissertacao/    <- Tabelas CSV estruturadas (Tabelas 03 a 12)
├── src/                        <- Código-fonte modular
│   ├── app/                    <- Aplicação web interativa (Streamlit)
│   └── core/                   <- Pipelines de dados, modelos e auditoria
└── tests/                      <- Suíte de testes automatizados
```

---

## 📄 Licença e Uso Acadêmico

Desenvolvido para fins de pesquisa acadêmica e modernização da gestão pública em restaurantes estudantis.  
Autor: **Michel Silva de Souza** ([micheldigui](https://github.com/micheldigui)).

