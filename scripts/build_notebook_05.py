import json
from pathlib import Path

nb = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# 05. Modelo Híbrido, Margens de Segurança e Avaliação Retrospectiva vs. Prospectiva\n',
                '**Artefato Analítico-Preditivo para Planejamento de Demanda e Mitigação do Desperdício**  \n',
                '*Mestrado Profissional em Educação Profissional e Tecnológica (ProfEPT) / Gestão Pública*  \n',
                '**Pesquisador:** Michel Silva de Souza  \n',
                '\n',
                '---\n',
                '\n',
                '## Storytelling: Da Predição Estatística à Tomada de Decisão na Gestão Pública\n',
                'Na gestão do restaurante institucional escolar, prever o consumo não é apenas um exercício de minimização de erro numérico: é uma **decisão de política pública** que equilibra dois imperativos constitucionais:\n',
                '1. **Segurança Alimentar e Nutricional dos Alunos:** Nenhum estudante que comparecer à escola pode ficar sem refeição (tolerância mínima ao déficit).\n',
                '2. **Eficiência e Zelo com o Erário Público:** Cada refeição produzida a mais e não consumida torna-se desperdício biológico e prejuízo financeiro (R$ 14,00 por refeição).\n',
                '\n',
                'Neste notebook, consolidamos os resultados em dois regimes temporais fundamentais:\n',
                '* **Período Retrospectivo (Fevereiro a Agosto de 2026 - $N=121$ dias):** Validação *Walk-Forward* prospectiva contínua (amostra de referência da pesquisa).\n',
                '* **Período Prospectivo (Setembro e Outubro de 2026 - $N=28$ dias):** Teste *out-of-time* cego em tempo real, cobrindo tanto a resiliência a eventos climáticos catastróficos (enchentes de setembro) quanto a estabilidade em operação padrão (outubro).\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import pandas as pd\n',
                'import numpy as np\n',
                'import matplotlib.pyplot as plt\n',
                'import seaborn as sns\n',
                'from pathlib import Path\n',
                'from IPython.display import Image, display\n',
                '\n',
                'from src.core.config import RESULTADOS\n',
                'from src.core.safety_margins import PASTA_TABELAS, PASTA_FIGURAS\n',
                '\n',
                'plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")\n',
                'plt.rcParams["font.sans-serif"] = "DejaVu Sans"\n',
                'plt.rcParams["figure.dpi"] = 140\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 1. Avaliação Retrospectiva (Fevereiro a Agosto de 2026 - $N=121$ dias)\n',
                'Abaixo apresentamos o confronto de modelos preditivos em $T+1$ (Preparo) e $T+2$ (Compras) frente ao especialista humano.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'tabela_04 = pd.read_csv(PASTA_TABELAS / "tabela_04_modelos_vs_nutricionista_t1_retrospectivo.csv", sep=";")\n',
                'print("=== TABELA 04: MODELOS PREDITIVOS VS. NUTRICIONISTA (T+1 - RETROSPECTIVO) ===")\n',
                'display(tabela_04[["Modelo / Origem", "Familia", "Cenario", "MAE", "RMSE", "WAPE_pct", "Sobra_Total_ref", "Deficit_Total_ref", "Taxa_Deficit_pct", "Custo_Potencial_Sobras_RS"]].head(10))\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'display(Image(filename=str(PASTA_FIGURAS / "figura_18_ablation_cenarios_horizontes_retrospectivo.png")))\n',
                'display(Image(filename=str(PASTA_FIGURAS / "figura_19_confronto_taxonomia_ensembles_retrospectivo.png")))\n',
                'display(Image(filename=str(PASTA_FIGURAS / "figura_20_modelos_vs_nutricionista_t1_retrospectivo.png")))\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '### 1.1. O Modelo Híbrido Adaptativo ($T+1$)\n',
                'Combinação linear convexa: $\\hat{y}_{\\text{híbrido}} = w \\cdot \\hat{y}_{\\text{IA}} + (1 - w) \\cdot \\hat{y}_{\\text{nutri}}$.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'display(Image(filename=str(PASTA_FIGURAS / "figura_21_fronteira_equilibrio_modelo_hibrido_retrospectivo.png")))\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '### 1.2. Margens Conformais de Segurança e Trade-off Operacional\n',
                'Adição de margens de resíduo empírico ($+10, +12, +15, +19$ refeições) para mitigar o déficit com mínimo acréscimo de sobras.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'tabela_05 = pd.read_csv(PASTA_TABELAS / "tabela_05_politicas_margens_retrospectivo.csv", sep=";")\n',
                'print("=== TABELA 05: AVALIAÇÃO DE POLÍTICAS E MARGENS CONFORMAIS (RETROSPECTIVO) ===")\n',
                'display(tabela_05[["Politica", "Margem", "MAE", "Sobra_Total_ref", "Deficit_Total_ref", "Dias_com_Deficit", "Taxa_Deficit_pct", "Custo_Potencial_Sobras_RS"]])\n',
                'display(Image(filename=str(PASTA_FIGURAS / "figura_22_margens_seguranca_conformal_retrospectivo.png")))\n',
                'display(Image(filename=str(PASTA_FIGURAS / "figura_23_tradeoff_sobras_deficits_custo_retrospectivo.png")))\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 2. Avaliação Prospectiva (Setembro e Outubro de 2026 - $N=28$ dias)\n',
                'O período de validação cega em tempo real, confrontando a resposta em dias regulares e durante o estresse climático severo de setembro.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'tabela_07 = pd.read_csv(PASTA_TABELAS / "tabela_07_desempenho_prospectivo_set_out.csv", sep=";")\n',
                'print("=== TABELA 07: DESEMPENHO NO BIMESTRE PROSPECTIVO (SET-OUT/2026) ===")\n',
                'display(tabela_07[["Configuracao", "Tipo", "MAE", "WAPE_pct", "Sobra_Total_ref", "Deficit_Total_ref", "Taxa_Deficit_pct", "Custo_Potencial_Sobras_RS"]])\n',
                'display(Image(filename=str(PASTA_FIGURAS / "figura_24_serie_temporal_bimestre_set_out_chuva_vs_consumo.png")))\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '### 2.1. Teste de Resiliência Climática: Dias Normais vs. Dias de Enchente\n',
                'Decomposição do mês de Setembro entre os 11 dias regulares e os 5 dias de enchente extrema na bacia do Alto Tietê.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'tabela_08 = pd.read_csv(PASTA_TABELAS / "tabela_08_resiliencia_climatica_setembro_enchentes.csv", sep=";")\n',
                'print("=== TABELA 08: RESILIÊNCIA CLIMÁTICA EM SETEMBRO DE 2026 ===")\n',
                'display(tabela_08[["Recorte", "Metodo", "MAE", "RMSE", "WAPE_pct", "Sobra_Total_ref", "Deficit_Total_ref", "Custo_Potencial_Sobras_RS"]])\n',
                'display(Image(filename=str(PASTA_FIGURAS / "figura_25_confronto_diario_setembro_enchentes_ia_vs_nutri.png")))\n',
                'display(Image(filename=str(PASTA_FIGURAS / "figura_26_validacao_outubro_estabilidade.png")))\n',
                'display(Image(filename=str(PASTA_FIGURAS / "figura_27_balanco_sobras_faltas_prospectivo.png")))\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 3. Síntese Comparativa Retrospectivo vs. Prospectivo\n',
                'Visão consolidada confrontando os resultados da pesquisa para a escrita do Capítulo 5 da Dissertação.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'tabela_10 = pd.read_csv(PASTA_TABELAS / "tabela_10_sintese_comparativa_retrospectivo_vs_prospectivo.csv", sep=";")\n',
                'print("=== TABELA 10: SÍNTESE COMPARATIVA DE REGIMES OPERACIONAIS ===")\n',
                'display(tabela_10)\n',
                'display(Image(filename=str(PASTA_FIGURAS / "figura_28_sintese_retrospectivo_vs_prospectivo.png")))\n'
            ]
        }
    ],
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'}
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

Path('notebooks/05_modelo_hibrido_e_margens_seguranca.ipynb').parent.mkdir(parents=True, exist_ok=True)
with open('notebooks/05_modelo_hibrido_e_margens_seguranca.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Notebook 05 gerado com sucesso!')
