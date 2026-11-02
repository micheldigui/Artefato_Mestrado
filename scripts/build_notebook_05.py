import json
from pathlib import Path

nb = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# 05. Avaliação Prospectiva (Set-Out/2026): Modelo Congelado em 31/08/2026\n',
                '**Artefato Analítico-Preditivo para Planejamento de Demanda e Mitigação do Desperdício**  \n',
                '*Mestrado Profissional em Educação Profissional e Tecnológica (ProfEPT) / Gestão Pública*  \n',
                '**Pesquisador:** Michel Silva de Souza  \n',
                '\n',
                '---\n',
                '\n',
                '## Storytelling: O Teste do Modelo no Período Seguinte\n',
                'Na gestão do restaurante institucional escolar, prever o consumo não é apenas um exercício de minimização de erro numérico: é uma **decisão de política pública** que equilibra dois imperativos constitucionais:\n',
                '1. **Segurança Alimentar e Nutricional dos Alunos:** Nenhum estudante que comparecer à escola pode ficar sem refeição (tolerância mínima ao déficit).\n',
                '2. **Eficiência e Zelo com o Erário Público:** Cada refeição produzida a mais e não consumida torna-se desperdício biológico e prejuízo financeiro (R$ 14,00 por refeição).\n',
                '\n',
                'Este notebook avalia o período seguinte ao usado para treinar e calibrar o modelo:\n',
                                '* **Período Prospectivo (Setembro e Outubro de 2026 - $N=33$ dias):** Modelos escolhidos em Fev-Ago e congelados em 31/08/2026, sem retreino, com métricas reprocessadas em 01/11/2026. O modelo é reconstruído com dados até 31/08, e sua especificação difere do ExtraTrees do torneio (`max_features` e início do treino em 2025).\n',
                '\n',
                '### Como os dias atípicos são tratados\n',
                'O registro de dia atípico só existe depois do consumo. Por isso, ele tem três usos distintos:\n',
                '1. **Treino:** o dia fica fora, para não ensinar um padrão que o modelo não tem como prever.\n',
                '2. **Lags das previsões seguintes:** dias com consumo muito baixo ficam fora, para não derrubar as semanas seguintes.\n',
                '3. **Avaliação:** o dia continua visível. O resultado principal usa todos os dias, porque a decisão de sexta não sabia do evento, e o recorte sem evento registrado aparece ao lado.\n'
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
                '## 1. Resultados de Setembro e Outubro de 2026 ($N=33$ dias)\n',
                'Resultado principal com o modelo congelado em todos os dias, ao lado do recorte sem evento registrado. O walk-forward aparece apenas como análise de sensibilidade.'
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
                'display(tabela_07[["Configuracao", "Regime", "Recorte", "MAE", "WAPE_pct", "Sobra_Total_ref", "Deficit_Total_ref", "Taxa_Deficit_pct", "Custo_Potencial_Sobras_RS"]])\n',
                'display(Image(filename=str(PASTA_FIGURAS / "figura_24_serie_temporal_bimestre_set_out_chuva_vs_consumo.png")))\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '### 1.1. Setembro: Dias sem Evento Registrado vs. Dias de Enchente\n',
                'Decomposição de setembro. Nenhum método antecipou as enchentes: as previsões continuaram muito acima do consumo observado. A lista de datas foi definida depois do fato.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'tabela_08 = pd.read_csv(PASTA_TABELAS / "tabela_08_resiliencia_climatica_setembro_enchentes.csv", sep=";")\n',
                'print("=== TABELA 08: SETEMBRO DE 2026, DIAS SEM EVENTO VS. ENCHENTES ===")\n',
                'display(tabela_08[["Recorte", "Metodo", "MAE", "RMSE", "WAPE_pct", "Sobra_Total_ref", "Deficit_Total_ref", "Custo_Potencial_Sobras_RS"]])\n',
                'display(pd.read_csv(PASTA_TABELAS / "tabela_08b_previsoes_diarias_enchentes.csv", sep=";"))\n',
                'display(Image(filename=str(PASTA_FIGURAS / "figura_25_confronto_diario_setembro_enchentes_ia_vs_nutri.png")))\n',
                'display(Image(filename=str(PASTA_FIGURAS / "figura_26_validacao_outubro_estabilidade.png")))\n',
                'display(Image(filename=str(PASTA_FIGURAS / "figura_27_balanco_sobras_faltas_prospectivo.png")))\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 2. Síntese Comparativa Retrospectivo vs. Prospectivo\n',
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
                'print("=== TABELA 10: SÍNTESE COMPARATIVA DOS PERÍODOS ===")\n',
                'display(tabela_10)\n',
                'display(Image(filename=str(PASTA_FIGURAS / "figura_28_sintese_retrospectivo_vs_prospectivo.png")))\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 3. Margens, Previsão de Clima e Comparação Estatística\n',
                'As margens foram calibradas em Fev-Ago e aplicadas sem recalibrar em Set-Out. O Cenário 3 usa somente previsões de clima arquivadas. A comparação entre métodos usa bootstrap pareado por semana: quando o intervalo de 95% inclui zero, não se declara vencedor.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'print("=== TABELA 09: MARGENS EMPÍRICAS E COBERTURA ===")\n',
                'display(pd.read_csv(PASTA_TABELAS / "tabela_09_margens_empiricas_quantis.csv", sep=";"))\n',
                'print("=== TABELA 11: CENÁRIO 3 COM O MODELO CONGELADO ===")\n',
                'display(pd.read_csv(PASTA_TABELAS / "tabela_11_cenario_3_clima_congelado.csv", sep=";")[["Recorte", "Metodo", "n_dias", "MAE", "RMSE", "WAPE_pct", "Sobra_Total_ref", "Deficit_Total_ref"]])\n',
                'print("=== TABELA 12: COMPARAÇÃO PAREADA POR SEMANA ===")\n',
                'display(pd.read_csv(PASTA_TABELAS / "tabela_12_comparacao_bootstrap_semanal.csv", sep=";"))\n'
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

