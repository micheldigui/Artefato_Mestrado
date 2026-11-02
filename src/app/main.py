"""
Interface Web Operacional do Artefato — Apoio à Decisão e Storytelling da Pesquisa
Pesquisa: Artefato Analítico-Preditivo para Planejamento da Produção de Refeições e Combate ao Desperdício
Autor / Mestrando: Michel Silva de Souza
Orientador: Prof. Dr. Carlos Hideo Arima
Coorientador: Prof. Dr. Rafael Nobre Orsi
Programa: Mestrado Profissional em Gestão e Tecnologia em Sistemas Produtivos (PPG-GTSP)
Campo de Aplicação Empírica: Restaurante Estudantil de Instituição Pública de Ensino
Metodologia: Design Science Research (DSR)
"""

import os
import sys
import json

app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Raiz do projeto
ROOT_DIR = os.path.abspath(os.path.join(app_dir, "..", ".."))

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import hmac
import html
from datetime import datetime, date, timedelta
import importlib

try:
    import storage
    importlib.reload(storage)
    from storage import (
        carregar_historico_consumo,
        salvar_consumo_diario,
        carregar_planejamento_nutricionista,
        salvar_planejamento_semanal,
        consultar_consumo_por_data,
        consultar_dados_completos_data,
        salvar_edicao_operacional_completa,
        listar_ultimas_edicoes,
        carregar_fato_diaria_completa,
        CSV_CONSUMO_ATUALIZADO
    )
    import pipeline
    importlib.reload(pipeline)
    from pipeline import (
        executar_pipeline_previsao,
        DIAS_SEMANA_PT,
        obter_proximos_dias_uteis,
        CENARIOS_DICT,
        MODELOS_RANKING,
        MARGENS_INFO,
        FERIADOS_FIXOS_BR,
        CENARIO_PADRAO,
        MARGEM_PADRAO,
        MODELO_HIBRIDO,
        MODELO_IA,
        obter_clima_emissao,
        simular_previsao_data
    )
    import clima
    importlib.reload(clima)
    from clima import obter_clima_data
except ImportError:
    from . import storage, pipeline, clima
    importlib.reload(storage)
    importlib.reload(pipeline)
    importlib.reload(clima)
    from .storage import (
        carregar_historico_consumo,
        salvar_consumo_diario,
        carregar_planejamento_nutricionista,
        salvar_planejamento_semanal,
        consultar_consumo_por_data,
        consultar_dados_completos_data,
        salvar_edicao_operacional_completa,
        listar_ultimas_edicoes,
        carregar_fato_diaria_completa,
        CSV_CONSUMO_ATUALIZADO
    )
    from .pipeline import (
        executar_pipeline_previsao,
        DIAS_SEMANA_PT,
        obter_proximos_dias_uteis,
        CENARIOS_DICT,
        MODELOS_RANKING,
        MARGENS_INFO,
        FERIADOS_FIXOS_BR,
        CENARIO_PADRAO,
        MARGEM_PADRAO,
        MODELO_HIBRIDO,
        MODELO_IA,
        obter_clima_emissao,
        simular_previsao_data
    )
    from .clima import obter_clima_data

# Configuração da página
st.set_page_config(
    page_title="Artefato Preditivo de Refeições — Apoio à Decisão",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Gerenciamento de Sessão e Credenciais de Operação
if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False
if "usuario_nome" not in st.session_state:
    st.session_state["usuario_nome"] = "Visitante (Acesso Público)"
if "usuario_perfil" not in st.session_state:
    st.session_state["usuario_perfil"] = "visitante"

def obter_credencial(chave, padrao):
    try:
        return st.secrets.get(chave, padrao)
    except Exception:
        return padrao

SENHA_NUTRI = str(obter_credencial("SENHA_NUTRI", ""))
SENHA_ADMIN = str(obter_credencial("SENHA_ADMIN", ""))


def senha_confere(digitada, esperada):
    """Sem senha configurada em st.secrets, a edição permanece bloqueada."""
    return bool(esperada) and hmac.compare_digest(digitada.encode("utf-8"), esperada.encode("utf-8"))

# ==============================================================================
# BARRA LATERAL FIXA (SIDEBAR): NAVEGAÇÃO PRINCIPAL E CONTROLES DISCRETOS
# ==============================================================================
OPCOES_NAVEGACAO = [
    "Dashboard",
    "Previsões",
    "Simulador de Cenários",
    "Compras & Suprimentos",
    "Cadastrar Dados",
    "Guia Prático"
]

ICONES_NAVEGACAO = {
    "Dashboard": "📊",
    "Previsões": "🔮",
    "Simulador de Cenários": "🧪",
    "Compras & Suprimentos": "📦",
    "Cadastrar Dados": "✍️",
    "Guia Prático": "📖"
}

if "pagina_ativa" not in st.session_state:
    st.session_state["pagina_ativa"] = "Dashboard"

tema_escolhido = "Claro"

# ==============================================================================
# DESIGN SYSTEM: TEMA CLARO OPERACIONAL FIXO
# ==============================================================================
chart_text = "#334155"
chart_grid = "#e2e8f0"
chart_primary = "#008744"
chart_accent = "#0284c7"

root_vars = """
:root {
    --bg-app: #f8fafc;
    --bg-sidebar: #ffffff;
    --bg-card: #ffffff;
    --bg-card-hover: #f1f5f9;
    --border-subtle: #e2e8f0;
    --border-strong: #cbd5e1;
    --text-primary: #0f172a;
    --text-secondary: #334155;
    --text-muted: #64748b;
    --primary-color: #008744;
    --primary-hover: #007038;
    --primary-subtle: rgba(0, 135, 68, 0.08);
    --accent-color: #0284c7;
    --accent-hover: #0369a1;
    --accent-subtle: rgba(2, 132, 199, 0.08);
    --success-color: #15803d;
    --success-subtle: #dcfce7;
    --warning-color: #b45309;
    --warning-subtle: #fef3c7;
    --danger-color: #dc2626;
    --danger-subtle: #fee2e2;
    --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.04);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07);
}
"""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

{root_vars}

:root {{
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', Consolas, Monaco, monospace;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
}}

/* 1. Tipografia Global Unificada: Inter nos textos, preservando estritamente as fontes de ícones */
html, body, .stApp,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4,
div[data-testid="stMarkdownContainer"] h5,
div[data-testid="stMarkdownContainer"] h6,
button, input, select, textarea,
div[data-baseweb="select"] span,
div[data-baseweb="input"] input,
div[data-testid="stWidgetLabel"] p,
div[data-testid="stRadio"] label p,
div[data-testid="stCheckbox"] label p,
div[data-testid="stExpander"] details summary p {{
    font-family: var(--font-sans) !important;
}}

/* Preservação estrita dos ícones do Streamlit (Material Symbols / Icons) */
[data-testid="stIconMaterial"],
[class*="material-symbols"],
[class*="material-icons"] {{
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    white-space: nowrap !important;
    line-height: 1 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
}}

/* Alinhamento flexível e espaçamento limpo no cabeçalho do expander */
div[data-testid="stExpander"] details summary {{
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 10px 14px !important;
}}

div[data-testid="stExpander"] details summary svg {{
    flex-shrink: 0 !important;
}}

div[data-testid="stExpander"] details summary p {{
    margin: 0 !important;
    padding: 0 !important;
    display: inline-block !important;
}}

/* Monospace apenas para números e métricas operacionais */
.kpi-val, .day-prod-number, [data-testid="stMetricValue"], [data-testid="stMetricValue"] *, code, pre, .font-mono {{
    font-family: var(--font-mono) !important;
}}

/* 2. Layout, Fundo e Barra Lateral */
body, .stApp, [data-testid="stAppViewContainer"] {{
    background-color: var(--bg-app) !important;
    color: var(--text-primary) !important;
}}

[data-testid="stSidebar"] {{
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-subtle) !important;
}}

/* 3. Títulos e Textos */
h1, h2, h3, h4, h5, h6,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4,
div[data-testid="stMarkdownContainer"] h5,
div[data-testid="stMarkdownContainer"] h6 {{
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.2px !important;
}}

div[data-testid="stMarkdownContainer"] > p,
div[data-testid="stMarkdownContainer"] > ul,
div[data-testid="stMarkdownContainer"] > ol,
div[data-testid="stMarkdownContainer"] li {{
    color: var(--text-secondary) !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}}

strong, b,
div[data-testid="stMarkdownContainer"] strong,
div[data-testid="stMarkdownContainer"] b {{
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}}

.stCaption, div[data-testid="stCaptionContainer"] p {{
    color: var(--text-muted) !important;
    font-size: 12px !important;
    line-height: 1.4 !important;
}}

/* 4. Rótulos de Formulários */
label,
label[data-testid="stWidgetLabel"],
label[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"],
div[data-testid="stWidgetLabel"] p,
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label p,
div[data-testid="stCheckbox"] label,
div[data-testid="stCheckbox"] label p {{
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}}

/* 5. Contêineres de Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm) !important;
}}

/* 6. Controles de Formulário e Menus Suspensos */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
input, select, textarea {{
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
}}

/* Redução sutil da fonte nos seletores e menus suspensos para evitar cortes de texto */
div[data-baseweb="select"],
div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
ul[data-testid="stSelectboxVirtualDropdown"] li,
ul[role="listbox"] li,
div[data-baseweb="popover"],
div[data-baseweb="popover"] * {{
    font-size: 11.5px !important;
}}

/* Garante que o menu popover suspenso abra com largura suficiente e sem cortar texto */
div[data-baseweb="popover"] {{
    min-width: fit-content !important;
    max-width: 95vw !important;
}}

div[data-baseweb="popover"],
div[data-baseweb="menu"],
div[role="listbox"],
li[role="option"] {{
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
}}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {{
    background-color: var(--bg-card-hover) !important;
    color: var(--primary-color) !important;
}}

div[data-baseweb="tag"] {{
    background-color: var(--primary-subtle) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
}}
div[data-baseweb="tag"] span {{
    color: var(--primary-color) !important;
    font-weight: 600 !important;
}}

/* 7. Botões Operacionais e de Formulários */
button[kind="primary"],
button[data-testid="baseButton-primary"],
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {{
    background-color: var(--primary-color) !important;
    color: #ffffff !important;
    border-radius: var(--radius-sm) !important;
    border: none !important;
    font-weight: 700 !important;
    transition: all 0.15s ease !important;
}}
button[kind="primary"] *,
button[data-testid="baseButton-primary"] *,
div.stButton > button *,
div[data-testid="stFormSubmitButton"] > button * {{
    color: #ffffff !important;
    font-weight: 700 !important;
}}
button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {{
    background-color: var(--primary-hover) !important;
    box-shadow: var(--shadow-sm) !important;
}}

/* 8. Sanfonas e Expanders */
div[data-testid="stExpander"] {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
}}
div[data-testid="stExpander"] details summary,
div[data-testid="stExpander"] details summary p,
div[data-testid="stExpander"] details summary span {{
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}}

/* 9. Métricas Streamlit */
div[data-testid="stMetric"] {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-left: 4px solid var(--primary-color) !important;
    border-radius: var(--radius-md) !important;
    padding: 12px 16px !important;
    box-shadow: var(--shadow-sm) !important;
}}
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {{
    color: var(--text-muted) !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}}
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {{
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 24px !important;
    font-weight: 800 !important;
}}

/* 10. Top Header Banner Institucional */
.top-header {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-left: 5px solid var(--primary-color) !important;
    border-radius: var(--radius-md);
    padding: 14px 18px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    box-shadow: var(--shadow-sm);
}}
.header-left {{
    display: flex;
    align-items: center;
    gap: 12px;
}}
.logo-badge {{
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
}}
.header-titles h1 {{
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.3px;
    margin: 0;
    line-height: 1.25;
    color: var(--text-primary) !important;
}}
.header-titles p {{
    font-size: 12px;
    margin: 3px 0 0 0;
    color: var(--text-muted) !important;
}}
.header-right {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}}
.badge-status {{
    font-size: 11.5px;
    font-weight: 700;
    padding: 5px 11px;
    border-radius: 9999px;
    background: var(--success-subtle) !important;
    color: var(--success-color) !important;
    border: 1px solid var(--border-subtle) !important;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.badge-academico {{
    font-size: 11px;
    font-weight: 700;
    padding: 5px 11px;
    border-radius: 9999px;
    background: var(--primary-subtle) !important;
    color: var(--primary-color) !important;
    border: 1px solid var(--border-subtle) !important;
}}

/* 11. KPI Row & Cards */
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}}
.kpi-card {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md);
    padding: 14px 16px;
    box-shadow: var(--shadow-sm);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}}
.kpi-card.b-primary {{ border-left: 4px solid var(--primary-color) !important; }}
.kpi-card.b-accent {{ border-left: 4px solid var(--accent-color) !important; }}
.kpi-card.b-success {{ border-left: 4px solid var(--success-color) !important; }}
.kpi-card.b-warning {{ border-left: 4px solid var(--warning-color) !important; }}
.kpi-title {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted) !important;
    margin-bottom: 4px;
}}
.kpi-val {{
    font-size: 24px;
    font-weight: 800;
    font-family: var(--font-mono) !important;
    color: var(--text-primary) !important;
    line-height: 1.2;
}}
.kpi-sub {{
    font-size: 11.5px;
    color: var(--text-secondary) !important;
    margin-top: 3px;
}}

/* 12. Grade de Cartões Diários de Produção */
.day-prod-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}}
.day-prod-card {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: 4px solid var(--primary-color) !important;
    border-radius: var(--radius-md);
    padding: 14px 12px;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.day-prod-card:hover {{
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}}
.day-prod-date {{
    font-size: 11.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted) !important;
}}
.day-prod-dish {{
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-primary) !important;
    margin: 6px 0;
    min-height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.day-prod-number {{
    font-size: 32px;
    font-weight: 800;
    font-family: var(--font-mono) !important;
    color: var(--primary-color) !important;
    line-height: 1.1;
    margin: 4px 0;
}}
.day-prod-badge {{
    display: inline-block;
    font-size: 10.5px;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 999px;
    background: var(--primary-subtle) !important;
    color: var(--primary-color) !important;
    border: 1px solid var(--border-subtle) !important;
    margin-top: 5px;
}}
.day-prod-sub {{
    font-size: 11px;
    color: var(--text-muted) !important;
    margin-top: 5px;
}}

/* 13. Cabeçalhos dos Cards Explicativos (Como Funciona) */
.expl-card-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding-bottom: 8px;
    margin-bottom: 10px;
    border-bottom: 1px solid var(--border-subtle);
}}
.expl-border-primary {{
    border-left: 3px solid var(--primary-color);
    padding-left: 8px;
}}
.expl-border-accent {{
    border-left: 3px solid var(--accent-color);
    padding-left: 8px;
}}
.expl-border-warning {{
    border-left: 3px solid var(--warning-color);
    padding-left: 8px;
}}
.expl-icon {{
    font-size: 16px;
}}
.expl-title {{
    font-size: 13.5px;
    font-weight: 700;
    color: var(--text-primary) !important;
    letter-spacing: -0.1px;
}}

/* 14. Responsividade Adaptativa e Fluida */
/* Telas Estreitas (<= 768px): Empilhamento de colunas de conteudo (preservando barras horizontais utilitarias) */
@media screen and (max-width: 768px) {{
    div[data-testid="stHorizontalBlock"]:not(:has(.header-clean)) {{
        flex-direction: column !important;
        flex-wrap: wrap !important;
        gap: 14px !important;
    }}
    div[data-testid="stHorizontalBlock"]:not(:has(.header-clean)) > div[data-testid="column"] {{
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
        margin-bottom: 8px !important;
    }}
}}

.header-clean {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-left: 5px solid var(--primary-color) !important;
    border-radius: var(--radius-md);
    padding: 10px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: var(--shadow-sm);
    gap: 12px;
    min-height: 48px;
    margin-bottom: 12px !important;
    box-sizing: border-box;
    width: 100% !important;
}}
.header-clean .header-left {{
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
}}
.header-clean .logo-badge {{
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    font-weight: 800;
    color: #ffffff !important;
    flex-shrink: 0;
}}
.header-clean .header-titles {{
    min-width: 0;
}}
.header-clean .header-titles h1 {{
    font-size: 15px;
    font-weight: 700;
    margin: 0;
    line-height: 1.2;
    color: var(--text-primary) !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.header-clean .header-titles p {{
    font-size: 11px;
    margin: 2px 0 0 0;
    color: var(--text-muted) !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.header-right-badges {{
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
}}

@media (max-width: 640px) {{
    .header-clean {{
        padding: 6px 10px;
    }}
    .header-clean .header-titles h1 {{
        font-size: 13.5px;
    }}
    .header-clean .header-titles p {{
        font-size: 10px;
    }}
    .header-right-badges {{
        display: none !important;
    }}
}}

/* Dispositivos Móveis e Telas Compactas (<= 640px) */
@media (max-width: 640px) {{
    .kpi-row {{
        grid-template-columns: 1fr 1fr !important;
        gap: 8px !important;
    }}
    .kpi-card {{
        padding: 10px 8px !important;
    }}
    .kpi-val {{
        font-size: 19px !important;
    }}
    .day-prod-grid {{
        grid-template-columns: 1fr !important;
        gap: 8px !important;
    }}
    .top-header {{
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 8px !important;
    }}
}}

/* 15. Storytelling Badges e Subtítulos */
.story-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 10.5px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 3px 10px;
    border-radius: 9999px;
    background: var(--primary-subtle);
    color: var(--primary-color);
    border: 1px solid var(--border-subtle);
    margin-bottom: 6px;
}}
.story-subtitle {{
    font-size: 13.5px;
    color: var(--text-muted);
    margin-top: -6px;
    margin-bottom: 16px;
    line-height: 1.45;
}}

/* 16. Ranking Badges do Cardápio */
.rank-badge {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 9px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
    line-height: 1.2;
}}
.rank-1 {{
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
}}
.rank-2 {{
    background: #e0f2fe;
    color: #0369a1;
    border: 1px solid #bae6fd;
}}
.rank-3 {{
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
}}
.rank-other {{
    background: var(--bg-secondary);
    color: var(--text-muted);
    border: 1px solid var(--border-subtle);
}}

/* Oculta 100% qualquer resquício de barra lateral, cabeçalhos ou menus desnecessários */
#MainMenu {{ visibility: hidden !important; display: none !important; }}
footer {{ visibility: hidden !important; display: none !important; }}
div[data-testid="stDecoration"] {{ visibility: hidden !important; display: none !important; }}
div[data-testid="stStatusWidget"] {{ visibility: hidden !important; display: none !important; }}
.stDeployButton {{ display: none !important; }}
[data-testid="manage-app-button"] {{ display: none !important; }}
a[href*="github.com"] {{ display: none !important; }}

/* Ocultação total e irrestrita da barra lateral nativa do Streamlit */
section[data-testid="stSidebar"],
div[data-testid="stSidebarCollapsedControl"],
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"],
#ru-fixed-menu-container {{
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    pointer-events: none !important;
}}

header[data-testid="stHeader"] {{
    display: none !important;
    height: 0 !important;
}}

/* Espaçamento superior ideal para navegação 100% no corpo da página */
.block-container {{
    padding-top: 1.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}}

@media (max-width: 991px) {{
    .block-container {{
        padding-top: 1rem !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
    }}
}}

/* Carrossel de Deslize Horizontal Suave (Swipe) para Navegacao */
div[data-testid="stPills"], 
div[data-testid="stSegmentedControl"],
div[data-testid="stRadio"] > div[role="radiogroup"] {{
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
    scroll-behavior: smooth !important;
    gap: 8px !important;
    padding: 2px 2px 8px 2px !important;
    margin-bottom: 16px !important;
    scrollbar-width: thin !important;
    scrollbar-color: #cbd5e1 transparent !important;
    width: 100% !important;
}}

div[data-testid="stPills"]::-webkit-scrollbar,
div[data-testid="stSegmentedControl"]::-webkit-scrollbar,
div[data-testid="stRadio"] > div[role="radiogroup"]::-webkit-scrollbar {{
    height: 4px !important;
}}
div[data-testid="stPills"]::-webkit-scrollbar-thumb,
div[data-testid="stSegmentedControl"]::-webkit-scrollbar-thumb,
div[data-testid="stRadio"] > div[role="radiogroup"]::-webkit-scrollbar-thumb {{
    background-color: #cbd5e1 !important;
    border-radius: 4px !important;
}}

div[data-testid="stPills"] button,
div[data-testid="stSegmentedControl"] button {{
    flex-shrink: 0 !important;
    white-space: nowrap !important;
    border-radius: 20px !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    border: 1px solid #cbd5e1 !important;
    background-color: #ffffff !important;
    color: #334155 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.15s ease !important;
}}

div[data-testid="stPills"] button:hover {{
    background-color: #f0fdf4 !important;
    border-color: #008744 !important;
    color: #008744 !important;
}}

div[data-testid="stPills"] button[aria-selected="true"] {{
    background-color: #008744 !important;
    color: #ffffff !important;
    border-color: #008744 !important;
    box-shadow: 0 2px 8px rgba(0, 135, 68, 0.25) !important;
}}

/* Forçar modo claro 100% */
html, body, .stApp, [data-testid="stAppViewContainer"] {{
    background-color: #f8fafc !important;
    color: #0f172a !important;
    color-scheme: light !important;
}}

</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CABEÇALHO INSTITUCIONAL SUPERIOR & NAVEGAÇÃO HORIZONTAL MODERNA
# ==============================================================================
st.markdown(
    '<div class="header-clean">'
    '<div class="header-left">'
    '<div class="logo-badge">RU</div>'
    '<div class="header-titles">'
    '<h1>Restaurante Estudantil — Painel de Apoio à Decisão</h1>'
    '<p>Mestrado Profissional em Gestão e Tecnologia em Sistemas Produtivos (PPG-GTSP) • DSR 2026</p>'
    '</div>'
    '</div>'
    '<div class="header-right-badges">'
    '<span class="badge-status">● Sistema Ativo</span>'
    '<span class="badge-academico">DSR / Mestrado PPG-GTSP 2026</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

# Barra Superior com Navegacao Principal (Abas em Linha Unica com Deslize Lateral / Swipe)
if hasattr(st, "pills"):
    sel_top = st.pills(
        "Navegação:",
        OPCOES_NAVEGACAO,
        format_func=lambda x: f"{ICONES_NAVEGACAO.get(x, '')} {x}",
        default=st.session_state["pagina_ativa"],
        key="nav_superior_pills",
        label_visibility="collapsed"
    )
elif hasattr(st, "segmented_control"):
    sel_top = st.segmented_control(
        "Navegação:",
        OPCOES_NAVEGACAO,
        format_func=lambda x: f"{ICONES_NAVEGACAO.get(x, '')} {x}",
        default=st.session_state["pagina_ativa"],
        key="nav_superior_pills",
        label_visibility="collapsed"
    )
else:
    idx_top = OPCOES_NAVEGACAO.index(st.session_state["pagina_ativa"]) if st.session_state["pagina_ativa"] in OPCOES_NAVEGACAO else 0
    sel_top = st.radio(
        "Navegação:",
        OPCOES_NAVEGACAO,
        format_func=lambda x: f"{ICONES_NAVEGACAO.get(x, '')} {x}",
        index=idx_top,
        horizontal=True,
        key="nav_superior_pills",
        label_visibility="collapsed"
    )

if sel_top and sel_top != st.session_state["pagina_ativa"]:
    st.session_state["pagina_ativa"] = sel_top
    st.rerun()

pagina_ativa = st.session_state["pagina_ativa"]

def renderizar_painel_explicativo(objetivo_rotina, como_foi_feito_ciencia, como_usar_passos):
    """
    Renderiza um painel retrátil claro, acolhedor e visual para explicar a tela
    à nutricionista, à gestora e à banca examinadora, mantendo a tela limpa.
    """
    with st.expander("Instruções e Fundamentação Científica desta Tela", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.markdown(
                    '<div class="expl-card-header expl-border-primary">'
                    '<span class="expl-badge" style="font-size: 11px; font-weight: 700; background: #e8f5e9; color: #1b5e20; padding: 2px 6px; border-radius: 4px;">OBJETIVO</span>'
                    '<span class="expl-title">Para que serve na sua rotina?</span>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.markdown(objetivo_rotina)
        with c2:
            with st.container(border=True):
                st.markdown(
                    '<div class="expl-card-header expl-border-accent">'
                    '<span class="expl-icon">️</span>'
                    '<span class="expl-title">Como foi construído?</span>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.markdown(como_foi_feito_ciencia)
        with c3:
            with st.container(border=True):
                st.markdown(
                    '<div class="expl-card-header expl-border-warning">'
                    '<span class="expl-icon"></span>'
                    '<span class="expl-title">Como usar no dia a dia?</span>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.markdown(como_usar_passos)

def renderizar_painel_transparencia_dados(df_horizonte, titulo_painel="Diagnóstico de Insumos e Transparência dos Dados", horizonte_rotulo="Semana t+1"):
    """
    Renderiza um painel visual e auditável que expõe com 100% de transparência
    se as previsões estão utilizando dados reais inseridos (cardápio, clima, nutrição, calendário)
    ou valores de contingência/padrão definidos pelo artefato (fallbacks seguros).
    """
    if df_horizonte.empty:
        return

    is_let = df_horizonte["is_dia_letivo"] == True
    total_letivos = sum(is_let)
    dias_fechados = df_horizonte[~is_let]

    # 1. Cardápio
    alerta_cardapio = None
    if total_letivos == 0:
        b_card = "⛔ Não Aplicável"
        cor_card = "#64748b"
        bg_card = "#f1f5f9"
        sub_card = "Sem atendimento previsto no período."
    else:
        df_let = df_horizonte[is_let]
        status_card_vals = df_let["status_cardapio"].tolist()
        num_oficiais = sum(1 for s in status_card_vals if "Oficial" in str(s))
        if num_oficiais == total_letivos:
            b_card = "✅ Oficial Cadastrado (100%)"
            cor_card = "#166534"
            bg_card = "#dcfce7"
            sub_card = "Cardápio inserido na íntegra. Calibração por atratividade ativa."
        elif num_oficiais == 0:
            b_card = "⚠️ Padrão Neutro (Não Cadastrado)"
            cor_card = "#92400e"
            bg_card = "#fef3c7"
            sub_card = "Cardápio não inserido. A previsão segue sem o efeito do prato."
            alerta_cardapio = (
                f"⚠️ **Atenção Operacional — Cardápio Não Cadastrado ({horizonte_rotulo}):** "
                "O cardápio oficial para este período ainda não foi inserido na base pelo refeitório. "
                "A previsão segue sem o efeito do prato. "
                "Ela usa o consumo recente e o calendário letivo. "
                "Para refinar o preparo com a atratividade do prato real, realize o cadastro na aba **✍️ Cadastrar Dados**."
            )
        else:
            b_card = f"⚠️ Parcial ({num_oficiais}/{total_letivos} dias)"
            cor_card = "#92400e"
            bg_card = "#fef3c7"
            sub_card = f"{num_oficiais} dias com prato oficial e {total_letivos - num_oficiais} com perfil neutro."
            alerta_cardapio = f"⚠️ **Cardápio Parcialmente Cadastrado ({horizonte_rotulo}):** Alguns dias estão sem prato oficial lançado e seguem sem o efeito do prato. Perfil escolar regular neutro."

    # 2. Clima (Open-Meteo API vs Fallback)
    alerta_clima = None
    status_clima_vals = df_horizonte["status_clima"].astype(str).tolist()
    tem_satelite = any("Satélite" in s or "Open-Meteo" in s for s in status_clima_vals)
    tem_desativado = all("Desativado" in s for s in status_clima_vals)

    if tem_desativado:
        b_clima = "⚪ Desativado no Cenário"
        cor_clima = "#64748b"
        bg_clima = "#f1f5f9"
        sub_clima = "Cenário experimental selecionado não utiliza variáveis meteorológicas."
    elif tem_satelite:
        b_clima = "✅ Satélite Open-Meteo Ativo"
        cor_clima = "#0369a1"
        bg_clima = "#e0f2fe"
        sub_clima = "Previsão do tempo obtida em tempo real via satélite para o campus."
    else:
        b_clima = "ℹ️ Mediana Histórica (Offline)"
        cor_clima = "#475569"
        bg_clima = "#f1f5f9"
        sub_clima = "API meteorológica indisponível. Artefato utilizando a mediana histórica segura."
        alerta_clima = (
            "ℹ️ **Conexão Meteorológica em Contingência Segura:** "
            "A API pública Open-Meteo está temporariamente inacessível ou offline. "
            "O sistema ativou automaticamente a mediana histórica dos últimos anos para o campus (24,0°C e 2,5 mm), "
            "mantendo a operação preditiva 100% contínua e sem falhas."
        )

    # 3. Calendário Acadêmico
    alerta_cal = None
    if len(dias_fechados) > 0:
        desc_fechados = ", ".join([f"{r['data'][:5]} ({r['evento_tipo']})" for _, r in dias_fechados.iterrows()])
        b_cal = f"⛔ {len(dias_fechados)} Dia Não Letivo / Fechado"
        cor_cal = "#991b1b"
        bg_cal = "#fee2e2"
        sub_cal = f"Refeitório fechado em: {desc_fechados}."
        alerta_cal = (
            f"⛔ **Feriado / Recesso Identificado no Período:** {desc_fechados}. "
            "O refeitório estará fechado e a previsão foi fixada em estritamente 0 refeições. "
            "Este dia foi automaticamente excluído da média diária operacional."
        )
    else:
        b_cal = f"✅ {total_letivos} Dias Letivos Regulares"
        cor_cal = "#166534"
        bg_cal = "#dcfce7"
        sub_cal = "Sem feriados ou recessos: refeitório em pleno funcionamento."

    # 4. Nutrição / Planejamento Humano
    if total_letivos == 0:
        b_nutri = "⛔ Não Aplicável"
        cor_nutri = "#64748b"
        bg_nutri = "#f1f5f9"
        sub_nutri = "Sem atendimento previsto."
    else:
        num_nutri = sum(df_horizonte[is_let]["estimativa_nutri"].notna())
        if num_nutri > 0:
            b_nutri = f"✅ Informada pela Gestão ({num_nutri}/{total_letivos})"
            cor_nutri = "#166534"
            bg_nutri = "#dcfce7"
            sub_nutri = "Quantitativo humano integrado ao modelo híbrido."
        else:
            b_nutri = "ℹ️ Não Informada (IA Autônoma)"
            cor_nutri = "#475569"
            bg_nutri = "#f1f5f9"
            sub_nutri = "Meta diária não lançada. Modelos operando de forma 100% autônoma."

    # Renderização do Painel HTML
    painel_html = f'''
    <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 14px 16px; margin-bottom: 16px; box-shadow: var(--shadow-sm);">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; border-bottom: 1px solid var(--border-subtle); padding-bottom: 8px;">
            <div style="font-size: 14px; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 6px;">
                <span>🔍</span> <span>{titulo_painel}</span>
            </div>
            <span style="font-size: 11px; font-weight: 600; color: var(--text-muted); background: #f1f5f9; padding: 2px 8px; border-radius: 4px;">Auditoria de Insumos & Proveniência</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px;">
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 12px;">
                <div style="font-size: 11.5px; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 6px;">🥗 Cardápio Escolar</div>
                <div style="font-size: 11px; font-weight: 700; color: {cor_card}; background: {bg_card}; padding: 3px 8px; border-radius: 4px; display: inline-block; margin-bottom: 6px;">{b_card}</div>
                <div style="font-size: 11px; color: #64748b; line-height: 1.35;">{sub_card}</div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 12px;">
                <div style="font-size: 11.5px; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 6px;">🌦️ Meteorologia (Clima)</div>
                <div style="font-size: 11px; font-weight: 700; color: {cor_clima}; background: {bg_clima}; padding: 3px 8px; border-radius: 4px; display: inline-block; margin-bottom: 6px;">{b_clima}</div>
                <div style="font-size: 11px; color: #64748b; line-height: 1.35;">{sub_clima}</div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 12px;">
                <div style="font-size: 11.5px; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 6px;">📅 Calendário Acadêmico</div>
                <div style="font-size: 11px; font-weight: 700; color: {cor_cal}; background: {bg_cal}; padding: 3px 8px; border-radius: 4px; display: inline-block; margin-bottom: 6px;">{b_cal}</div>
                <div style="font-size: 11px; color: #64748b; line-height: 1.35;">{sub_cal}</div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 12px;">
                <div style="font-size: 11.5px; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 6px;">👩‍🍳 Meta da Nutrição</div>
                <div style="font-size: 11px; font-weight: 700; color: {cor_nutri}; background: {bg_nutri}; padding: 3px 8px; border-radius: 4px; display: inline-block; margin-bottom: 6px;">{b_nutri}</div>
                <div style="font-size: 11px; color: #64748b; line-height: 1.35;">{sub_nutri}</div>
            </div>
        </div>
    </div>
    '''
    st.markdown(painel_html, unsafe_allow_html=True)

    if alerta_cal:
        st.info(alerta_cal)
    if alerta_cardapio:
        st.warning(alerta_cardapio)
    if alerta_clima:
        st.info(alerta_clima)


# ==============================================================================
# PÁGINA 1: DASHBOARD (SEÇÕES DINÂMICAS E FILTROS INTERATIVOS)
# ==============================================================================
if pagina_ativa == "Dashboard":
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
            "1. **Use os filtros interativos** para recortar anos ou dias da semana desejados.\n"
            "2. **Selecione uma das 5 seções temáticas** abaixo para navegar pelos gráficos interativos.\n"
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
            opcoes_filtro_periodo = [
                "Todo o Histórico (2023–2026)",
                "Reservas Discentes (2023–2025)",
                "Regime Nutricionista (2026 — Ano Corrente)",
                "Ano Letivo 2025",
                "Ano Letivo 2024",
                "Ano Letivo 2023",
                "Personalizado (Marcar Anos Específicos...)"
            ]
            filtro_ano = st.selectbox(
                "Filtrar por Período / Regime / Ano Letivo:",
                opcoes_filtro_periodo,
                index=0,
                help="Selecione 'Reservas Discentes (2023–2025)' para focar a EDA na fase em que o preparo dependia do agendamento voluntário dos estudantes."
            )
            
            anos_custom = None
            if filtro_ano == "Personalizado (Marcar Anos Específicos...)":
                anos_custom = st.multiselect(
                    "Marque os anos que deseja incluir:",
                    options=[2023, 2024, 2025, 2026],
                    default=[2023, 2024, 2025],
                    format_func=lambda a: f"{a} (Reservas)" if a < 2026 else f"{a} (Nutricionista)"
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
    if filtro_ano.startswith("Todo"):
        rotulo_filtro = "Todo o Histórico (2023–2026)"
    elif "Reservas Discentes" in filtro_ano:
        df_dash = df_dash[df_dash["ano"].isin([2023, 2024, 2025])]
        rotulo_filtro = "Reservas Discentes (2023–2025)"
    elif "2026" in filtro_ano:
        df_dash = df_dash[df_dash["ano"] == 2026]
        rotulo_filtro = "Regime Nutricionista (2026)"
    elif "2025" in filtro_ano:
        df_dash = df_dash[df_dash["ano"] == 2025]
        rotulo_filtro = "Ano Letivo 2025"
    elif "2024" in filtro_ano:
        df_dash = df_dash[df_dash["ano"] == 2024]
        rotulo_filtro = "Ano Letivo 2024"
    elif "2023" in filtro_ano:
        df_dash = df_dash[df_dash["ano"] == 2023]
        rotulo_filtro = "Ano Letivo 2023"
    elif filtro_ano == "Personalizado (Marcar Anos Específicos...)":
        if anos_custom:
            df_dash = df_dash[df_dash["ano"].isin(anos_custom)]
            rotulo_filtro = f"Anos: {', '.join(str(a) for a in sorted(anos_custom))}"
        else:
            df_dash = df_dash.iloc[0:0]
            rotulo_filtro = "Nenhum ano selecionado"
    else:
        rotulo_filtro = "Todo o Histórico (2023–2026)"
        
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
                st.caption(f"Barras acima de zero indicam sobras; barras abaixo indicam déficit no período: **{rotulo_filtro}**.")
                
                fig_desv = px.bar(
                    df_reg,
                    x="data_dt",
                    y="desvio",
                    color="tipo_desvio",
                    color_discrete_map={"Sobra / Excedente (+)": "#f59e0b", "Falta Potencial (-)": "#ef4444", "Equilíbrio Exato": "#94a3b8"},
                    labels={"data_dt": "Data", "desvio": "Saldo (Planejado - Real)", "tipo_desvio": "Balanço"},
                    hover_data={"data": True, "consumo": True, "referencia_planejada": True, "desvio": True, "data_dt": False}
                )
                fig_desv.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_title=None,
                    yaxis_title="Diferença (ref)"
                )
                st.plotly_chart(fig_desv, use_container_width=True)
                
            with col_r2:
                st.markdown("##### Linha do Tempo: Demanda Real vs. Quantitativo Planejado")
                st.caption(f"Consumo real (linha verde) vs quantitativo planejado/reservas (linha âmbar) no período: **{rotulo_filtro}**.")
                
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(
                    x=df_reg["data_dt"], y=df_reg["consumo"],
                    mode="lines+markers", name="Consumo Real",
                    line=dict(color="#10b981", width=2.5),
                    marker=dict(size=5, color="#10b981"),
                    hovertemplate="Data: %{x|%d/%m/%Y}<br>Consumo Real: %{y:.0f} ref<extra></extra>"
                ))
                fig_time.add_trace(go.Scatter(
                    x=df_reg["data_dt"], y=df_reg["referencia_planejada"],
                    mode="lines", name="Planejado / Reserva",
                    line=dict(color="#f59e0b", width=2, dash="dash"),
                    hovertemplate="Data: %{x|%d/%m/%Y}<br>Planejado: %{y:.0f} ref<extra></extra>"
                ))
                fig_time.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_title=None,
                    yaxis_title="Total de Refeições"
                )
                st.plotly_chart(fig_time, use_container_width=True)

            # Linha 2 de Análises: Balanço Anual de Falta vs Sobra e Dispersão Interativa 45°
            col_r3, col_r4 = st.columns(2)
            with col_r3:
                st.markdown("##### Potencial de Falta (Déficit) vs. Sobra por Ano")
                st.caption("Comparativo anual consolidado de refeições faltantes vs. excedentes acumuladas (Dissertação Fig. 02).")
                
                df_bal_base = df_dash.copy() if len(df_dash) > 0 else df_validos_geral.copy()
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
                
                anos_list = [str(a) for a in df_bal_ano["ano"]]
                sobras_list = [float(v) for v in df_bal_ano["sobra_ref"]]
                faltas_list = [float(v) for v in df_bal_ano["falta_ref"]]
                max_vol = max(max(sobras_list or [0]), max(faltas_list or [0]))

                fig_bal = go.Figure()
                fig_bal.add_trace(go.Bar(
                    x=anos_list,
                    y=sobras_list,
                    name="Sobras Potenciais (Excedente)",
                    text=[f"{v:,.0f} ref" for v in sobras_list],
                    textposition="outside",
                    marker_color="#f59e0b",
                    cliponaxis=False
                ))
                fig_bal.add_trace(go.Bar(
                    x=anos_list,
                    y=faltas_list,
                    name="Déficit Potencial (Falta no Agendamento)",
                    text=[f"{v:,.0f} ref" for v in faltas_list],
                    textposition="outside",
                    marker_color="#ef4444",
                    cliponaxis=False
                ))
                fig_bal.update_layout(
                    barmode="group",
                    height=280,
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_title=None,
                    yaxis=dict(title="Volume Acumulado", range=[0, max_vol * 1.18 if max_vol > 0 else 100])
                )
                st.plotly_chart(fig_bal, use_container_width=True)
                
            with col_r4:
                st.markdown("##### Dispersão: Consumo Real vs. Referência Planejada")
                st.caption("Pontos próximos da linha diagonal tracejada indicam alta precisão operacional.")
                
                df_disp = df_dash.copy()
                c_n = df_disp["planejado_nutri"] if "planejado_nutri" in df_disp.columns else pd.Series(index=df_disp.index, dtype=float)
                c_r = df_disp["reservas"] if "reservas" in df_disp.columns else pd.Series(index=df_disp.index, dtype=float)
                df_disp["planejado"] = c_n.combine_first(c_r)
                df_disp = df_disp.dropna(subset=["planejado", "consumo"]).copy()
                df_disp["regime_nome"] = df_disp["ano"].apply(lambda a: "Nutricionista (2026)" if a == 2026 else "Reservas Alunos (2023-2025)")
                
                fig_disp = px.scatter(
                    df_disp,
                    x="planejado",
                    y="consumo",
                    color="regime_nome",
                    color_discrete_map={"Nutricionista (2026)": "#008744", "Reservas Alunos (2023-2025)": "#0284c7"},
                    labels={"planejado": "Planejado / Reserva (ref)", "consumo": "Consumo Real (ref)", "regime_nome": "Regime"},
                    hover_data={"data": True, "consumo": True, "planejado": True}
                )
                max_v = max(df_disp["planejado"].max(), df_disp["consumo"].max()) if len(df_disp) > 0 else 300
                fig_disp.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(color="#94a3b8", dash="dash", width=1.5))
                fig_disp.update_traces(marker=dict(size=7, opacity=0.65))
                fig_disp.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_disp, use_container_width=True)

            # Expander com Figuras Acadêmicas Oficiais da Dissertação
            with st.expander("🖼️ Ver Figuras Acadêmicas da Dissertação (Figuras 01 e 02)", expanded=False):
                col_f_a, col_f_b = st.columns(2)
                with col_f_a:
                    p1 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_01_confronto_metodos_humanos.png")
                    if os.path.exists(p1):
                        st.image(p1, caption="Figura 01: Confronto dos Métodos Humanos (Reservas Discentes vs. Nutricionista)", use_container_width=True)
                with col_f_b:
                    p2 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_02_balanco_sobra_falta_metodos.png")
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
            
            def map_dia_pt_local(val):
                val_s = str(val).lower()
                if "seg" in val_s: return "Segunda"
                if "ter" in val_s: return "Terça"
                if "qua" in val_s: return "Quarta"
                if "qui" in val_s: return "Quinta"
                if "sex" in val_s: return "Sexta"
                return None
            
            meses_map = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun", 7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}
            
            df_dash_d = df_dash.copy()
            df_dash_d["dia_nome"] = df_dash_d["dia_da_semana"].apply(map_dia_pt_local)
            df_d = df_dash_d[df_dash_d["dia_nome"].notna()].groupby("dia_nome")["consumo"].agg(["count", "mean"]).reset_index()
            df_d["consumo_medio"] = df_d["mean"].round(1)
            
            # Linha 1: Média por Dia da Semana e Sazonalidade Mensal
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("##### Média de Presença por Dia da Semana")
                st.caption("Comportamento médio da frequência dos estudantes de segunda a sexta-feira.")
                
                ordem_dias = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]
                df_d_sorted = df_d.set_index("dia_nome").reindex(ordem_dias).dropna(subset=["consumo_medio"]).reset_index()
                max_yd = float(df_d_sorted["consumo_medio"].max()) if not df_d_sorted.empty else 160.0
                
                fig_d = go.Figure(go.Bar(
                    x=df_d_sorted["dia_nome"].tolist(),
                    y=df_d_sorted["consumo_medio"].tolist(),
                    text=[f"{v:.1f} ref" for v in df_d_sorted["consumo_medio"]],
                    textposition="outside",
                    marker_color="#008744",
                    customdata=df_d_sorted["count"].tolist(),
                    hovertemplate="<b>Dia: %{x}</b><br>Média: %{y:.1f} refeições<br>Amostra: %{customdata} datas<extra></extra>",
                    cliponaxis=False
                ))
                fig_d.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=25, b=10),
                    xaxis=dict(title=None, categoryorder="array", categoryarray=ordem_dias),
                    yaxis=dict(title="Média de Refeições Servidas", range=[0, max_yd * 1.18]),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_d, use_container_width=True)
                
            with col_d2:
                st.markdown("##### Sazonalidade ao Longo dos Meses do Ano Letivo")
                st.caption("Picos no meio dos semestres (março/agosto) e quedas típicas em inícios de ano letivo e períodos de recesso.")
                
                df_dash_mes = df_dash.copy()
                df_dash_mes["mes_nome"] = df_dash_mes["mes"].map(meses_map)
                df_m = df_dash_mes.groupby(["mes", "mes_nome"])["consumo"].agg(["count", "mean"]).reset_index().sort_values("mes")
                df_m["consumo_medio"] = df_m["mean"].round(1)
                df_m_clean = df_m.dropna(subset=["consumo_medio"]).copy()
                
                ordem_meses = [m for m in meses_map.values() if m in df_m_clean["mes_nome"].values]
                max_ym = float(df_m_clean["consumo_medio"].max()) if not df_m_clean.empty else 160.0
                
                fig_m = go.Figure(go.Bar(
                    x=df_m_clean["mes_nome"].tolist(),
                    y=df_m_clean["consumo_medio"].tolist(),
                    text=[f"{v:.1f} ref" for v in df_m_clean["consumo_medio"]],
                    textposition="outside",
                    marker_color="#0284c7",
                    customdata=df_m_clean["count"].tolist(),
                    hovertemplate="<b>Mês: %{x}</b><br>Média: %{y:.1f} refeições<br>Amostra: %{customdata} datas<extra></extra>",
                    cliponaxis=False
                ))
                fig_m.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=25, b=10),
                    xaxis=dict(title=None, categoryorder="array", categoryarray=ordem_meses),
                    yaxis=dict(title="Média de Refeições Servidas", range=[0, max_ym * 1.18]),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_m, use_container_width=True)

            # Linha 2: Desmobilização de Fim de Semestre e Curva Semanal por Ano
            col_d3, col_d4 = st.columns(2)
            with col_d3:
                st.markdown("##### Desmobilização Acadêmica no Fim do Semestre")
                st.caption("Queda expressiva de consumo nos últimos 15 dias letivos antes das férias (Dissertação Fig. 05).")
                
                df_fim = df_validos_geral.dropna(subset=["dias_ate_fim_bloco"]).copy()
                df_fim = df_fim[df_fim["dias_ate_fim_bloco"] <= 15]
                df_fim_agg = df_fim.groupby("dias_ate_fim_bloco")["consumo"].mean().reset_index()
                df_fim_agg["consumo_medio"] = df_fim_agg["consumo"].round(1)
                df_fim_agg["dias_restantes"] = df_fim_agg["dias_ate_fim_bloco"].astype(int)
                
                fig_fim = px.line(
                    df_fim_agg.sort_values("dias_restantes", ascending=False),
                    x="dias_restantes",
                    y="consumo_medio",
                    markers=True,
                    labels={"dias_restantes": "Dias Restantes até Férias", "consumo_medio": "Consumo Médio (ref)"},
                    color_discrete_sequence=["#dc2626"]
                )
                fig_fim.update_xaxes(autorange="reversed")
                fig_fim.update_traces(line=dict(width=2.5), marker=dict(size=7))
                fig_fim.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=25, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis_title="Consumo Médio (ref)"
                )
                st.plotly_chart(fig_fim, use_container_width=True)
                
            with col_d4:
                st.markdown("##### Curva Semanal Comparativa por Ano (2023–2026)")
                st.caption("Comprova que a queda de sexta-feira é estrutural e se repetiu em todos os 4 anos analisados.")
                
                df_sem_ano = df_dash.copy() if len(df_dash["ano"].unique()) > 1 else df_validos_geral.copy()
                df_sem_ano["dia_nome"] = df_sem_ano["dia_da_semana"].apply(map_dia_pt_local)
                df_sem_ano = df_sem_ano[df_sem_ano["dia_nome"].notna()]
                df_s_agg = df_sem_ano.groupby(["ano", "dia_nome"])["consumo"].mean().reset_index()
                df_s_agg["consumo_medio"] = df_s_agg["consumo"].round(1)
                df_s_agg["ano_str"] = df_s_agg["ano"].astype(str)
                
                fig_sem = px.line(
                    df_s_agg,
                    x="dia_nome",
                    y="consumo_medio",
                    color="ano_str",
                    markers=True,
                    category_orders={"dia_nome": ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]},
                    color_discrete_map={"2023": "#0284c7", "2024": "#f59e0b", "2025": "#10b981", "2026": "#dc2626"},
                    labels={"dia_nome": "Dia da Semana", "consumo_medio": "Média de Refeições", "ano_str": "Ano"}
                )
                fig_sem.update_traces(line=dict(width=2.2), marker=dict(size=6))
                fig_sem.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=25, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_title=None,
                    yaxis_title="Média de Refeições"
                )
                st.plotly_chart(fig_sem, use_container_width=True)

            with st.expander("🖼️ Ver Figuras Acadêmicas da Dissertação (Figuras 03, 04 e 05)", expanded=False):
                col_f_c1, col_f_c2, col_f_c3 = st.columns(3)
                with col_f_c1:
                    p3 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_03_perfil_semanal_por_ano.png")
                    if os.path.exists(p3):
                        st.image(p3, caption="Figura 03: Perfil Semanal por Ano Letivo", use_container_width=True)
                with col_f_c2:
                    p4 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_04_sazonalidade_mensal.png")
                    if os.path.exists(p4):
                        st.image(p4, caption="Figura 04: Sazonalidade ao Longo dos Meses", use_container_width=True)
                with col_f_c3:
                    p5 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_05_desmobilizacao_fim_semestre.png")
                    if os.path.exists(p5):
                        st.image(p5, caption="Figura 05: Desmobilização Acadêmica no Fim do Semestre", use_container_width=True)

    # --------------------------------------------------------------------------
    # SUBSEÇÃO 3: CARDÁPIO & ATRATIVIDADE
    # --------------------------------------------------------------------------
    elif "3. Cardápio" in secao_dash:
        with st.container(border=True):
            st.markdown("#### 🍗 Influência do Cardápio: Tipo de Carne e Método Culinário")
            st.write(
                "**O prato servido dita o apetite discente:** "
                "Alunos respondem diretamente ao prato principal anunciado no mural. "
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
                st.caption(f"Comparativo visual de consumo médio no período: **{rotulo_filtro}**.")
                
                max_yp = float(df_p_agg["consumo_medio"].max()) if not df_p_agg.empty else 160.0
                fig_p = go.Figure(go.Bar(
                    x=df_p_agg["proteina_nome"].tolist(),
                    y=df_p_agg["consumo_medio"].tolist(),
                    text=[f"{v:.1f} ref" for v in df_p_agg["consumo_medio"]],
                    textposition="outside",
                    marker_color="#008744",
                    customdata=df_p_agg["count"].tolist(),
                    hovertemplate="<b>Carne: %{x}</b><br>Média: %{y:.1f} ref<br>Amostra: %{customdata} datas<extra></extra>",
                    cliponaxis=False
                ))
                fig_p.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=25, b=10),
                    xaxis=dict(title=None),
                    yaxis=dict(title="Média de Refeições Servidas", range=[0, max_yp * 1.18]),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_p, use_container_width=True)
                
            with col_p2:
                st.markdown(f"##### Ranking e Recomendações por Prato — {rotulo_filtro}")
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
                    '<div style="overflow-x: auto; border: 1px solid var(--border-subtle); border-radius: 8px;">'
                    '<table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: left;">'
                    '<thead><tr style="background: rgba(0,0,0,0.02); border-bottom: 2px solid var(--border-subtle);">'
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
                
                max_yprep = float(df_prep_agg["consumo_medio"].max()) if not df_prep_agg.empty else 160.0
                fig_prep = go.Figure(go.Bar(
                    x=df_prep_agg["preparo_nome"].tolist(),
                    y=df_prep_agg["consumo_medio"].tolist(),
                    text=[f"{v:.1f} ref" for v in df_prep_agg["consumo_medio"]],
                    textposition="outside",
                    marker_color="#0284c7",
                    customdata=df_prep_agg["count"].tolist(),
                    hovertemplate="<b>Técnica: %{x}</b><br>Média: %{y:.1f} ref<br>Amostra: %{customdata} datas<extra></extra>",
                    cliponaxis=False
                ))
                fig_prep.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=25, b=10),
                    xaxis=dict(title=None),
                    yaxis=dict(title="Média de Refeições Servidas", range=[0, max_yprep * 1.18]),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_prep, use_container_width=True)

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
                            "Mín": f"{v_mi:.0f} ref",
                            "Média": f"{v_me:.0f} ref",
                            "Máx": f"{v_ma:.0f} ref",
                            "Oscilação": f"±{amp:.0f} ref",
                            "Classificação Operacional": diag
                        })
                st.dataframe(pd.DataFrame(resumo_var), use_container_width=True, hide_index=True)

            with st.expander("🖼️ Ver Figuras Acadêmicas da Dissertação (Figuras 06 e 07)", expanded=False):
                col_f_p1, col_f_p2 = st.columns(2)
                with col_f_p1:
                    p6 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_06_oferta_consumo_por_proteina.png")
                    if os.path.exists(p6):
                        st.image(p6, caption="Figura 06: Frequência de Oferta vs Consumo Médio por Proteína", use_container_width=True)
                with col_f_p2:
                    p7 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_07_consumo_por_tecnica_preparo.png")
                    if os.path.exists(p7):
                        st.image(p7, caption="Figura 07: Impacto da Técnica Culinária na Presença Discente", use_container_width=True)

    # --------------------------------------------------------------------------
    # SUBSEÇÃO 4: CLIMA & EFEITO METEOROLÓGICO
    # --------------------------------------------------------------------------
    elif "4. Clima" in secao_dash:
        with st.container(border=True):
            st.markdown("#### 🌧️ Efeito Meteorológico: Chuva e Temperatura")
            st.write(
                "**A sensibilidade climática da comunidade acadêmica:** "
                "Ao contrário de escritórios convencionais, estudantes transitam a pé pelo campus. "
                "Em dias secos, a presença atinge seu pico regular; chuvas fortes e tempestades (>15 mm) provocam queda imediata de presença, "
                "exigindo recuo preventivo na produção para evitar desperdício de panelas cheias."
            )
            
            df_ch = df_dash.dropna(subset=["precipitation_sum"]).copy()
            def categorizar_chuva_local(mm):
                if mm == 0: return "Dia Seco (0 mm)"
                elif mm <= 5.0: return "Garoa (0.1 a 5 mm)"
                elif mm <= 15.0: return "Chuva Forte (5.1 a 15 mm)"
                else: return "Tempestade (>15 mm)"
                
            df_ch["faixa_chuva"] = df_ch["precipitation_sum"].apply(categorizar_chuva_local)
            ordem_chuva = ["Dia Seco (0 mm)", "Garoa (0.1 a 5 mm)", "Chuva Forte (5.1 a 15 mm)", "Tempestade (>15 mm)"]
            df_ch_agg = df_ch.groupby("faixa_chuva", observed=False)["consumo"].agg(["count", "mean"]).reindex(ordem_chuva).dropna().reset_index()
            df_ch_agg["consumo_medio"] = df_ch_agg["mean"].round(1)
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("##### Consumo Médio por Intensidade de Chuva")
                st.caption("Queda acentuada de alunos nos dias de tempestade (>15 mm).")
                
                max_ych = float(df_ch_agg["consumo_medio"].max()) if not df_ch_agg.empty else 160.0
                fig_ch = go.Figure(go.Bar(
                    x=df_ch_agg["faixa_chuva"].tolist(),
                    y=df_ch_agg["consumo_medio"].tolist(),
                    text=[f"{v:.1f} ref" for v in df_ch_agg["consumo_medio"]],
                    textposition="outside",
                    marker_color="#0284c7",
                    customdata=df_ch_agg["count"].tolist(),
                    hovertemplate="<b>Chuva: %{x}</b><br>Média: %{y:.1f} ref<br>Amostra: %{customdata} datas<extra></extra>",
                    cliponaxis=False
                ))
                fig_ch.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=25, b=10),
                    xaxis=dict(title=None, categoryorder="array", categoryarray=ordem_chuva),
                    yaxis=dict(title="Média de Refeições Servidas", range=[0, max_ych * 1.18]),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_ch, use_container_width=True)
                
            with col_c2:
                st.markdown("##### Relação Chuva (mm) vs. Refeições Servidas")
                st.caption("Cada ponto representa um dia letivo real; a linha vermelha mostra a tendência.")
                
                fig_ch_scatter = px.scatter(
                    df_ch,
                    x="precipitation_sum",
                    y="consumo",
                    labels={"precipitation_sum": "Volume de Chuva (mm)", "consumo": "Refeições Servidas"},
                    color_discrete_sequence=["#0284c7"],
                    hover_data={"data": True, "consumo": True, "precipitation_sum": True}
                )
                fig_ch_scatter.update_traces(marker=dict(size=6, opacity=0.6))
                
                # Linha de tendência linear calculada diretamente com NumPy (elimina dependência de statsmodels)
                if len(df_ch) > 1:
                    x_v = df_ch["precipitation_sum"].values.astype(float)
                    y_v = df_ch["consumo"].values.astype(float)
                    mask = ~np.isnan(x_v) & ~np.isnan(y_v)
                    if mask.sum() > 1:
                        slope, intercept = np.polyfit(x_v[mask], y_v[mask], 1)
                        x_line = np.array([float(x_v[mask].min()), float(x_v[mask].max())])
                        y_line = slope * x_line + intercept
                        fig_ch_scatter.add_trace(go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            name="Tendência Linear",
                            line=dict(color="#ef4444", width=2.5),
                            hoverinfo="skip"
                        ))
                
                fig_ch_scatter.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=25, b=10),
                    showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_ch_scatter, use_container_width=True)

            col_c3, col_c4 = st.columns(2)
            with col_c3:
                st.markdown("##### Frequência Discente por Faixa de Temperatura")
                st.caption("Adesão de estudantes conforme a temperatura máxima registrada.")
                
                def categorizar_temp_local(t):
                    if t < 20.0: return "Frio (<20°C)"
                    elif t <= 25.0: return "Agradável (20 a 25°C)"
                    elif t <= 30.0: return "Calor Moderado (25 a 30°C)"
                    else: return "Calor Intenso (>30°C)"
                
                df_ch["faixa_temp"] = df_ch["temperature_2m_max"].apply(categorizar_temp_local)
                ordem_temp = ["Frio (<20°C)", "Agradável (20 a 25°C)", "Calor Moderado (25 a 30°C)", "Calor Intenso (>30°C)"]
                df_t_agg = df_ch.groupby("faixa_temp", observed=False)["consumo"].agg(["count", "mean"]).reindex(ordem_temp).dropna().reset_index()
                df_t_agg["consumo_medio"] = df_t_agg["mean"].round(1)
                
                max_yt = float(df_t_agg["consumo_medio"].max()) if not df_t_agg.empty else 160.0
                fig_temp = go.Figure(go.Bar(
                    x=df_t_agg["faixa_temp"].tolist(),
                    y=df_t_agg["consumo_medio"].tolist(),
                    text=[f"{v:.1f} ref" for v in df_t_agg["consumo_medio"]],
                    textposition="outside",
                    marker_color="#f59e0b",
                    customdata=df_t_agg["count"].tolist(),
                    hovertemplate="<b>Temperatura: %{x}</b><br>Média: %{y:.1f} ref<br>Amostra: %{customdata} datas<extra></extra>",
                    cliponaxis=False
                ))
                fig_temp.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=25, b=10),
                    xaxis=dict(title=None, categoryorder="array", categoryarray=ordem_temp),
                    yaxis=dict(title="Média de Refeições Servidas", range=[0, max_yt * 1.18]),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_temp, use_container_width=True)
                
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
                    p8 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_08_sensibilidade_faixas_chuva.png")
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
            df_tor["Cor"] = df_tor["Correlacao"].apply(lambda c: "#008744" if c > 0 else "#ef4444")
            df_tor = df_tor.sort_values("Correlacao", ascending=True)
            
            col_t1, col_t2 = st.columns([1.2, 0.8])
            with col_t1:
                st.markdown("##### Tornado Chart: Correlação dos Atributos com o Consumo Real")
                st.caption("Verde indica correlação positiva (aumenta o consumo); Vermelho indica efeito redutor.")
                
                fig_tor = go.Figure(go.Bar(
                    x=df_tor["Correlacao"].tolist(),
                    y=df_tor["Atributo"].tolist(),
                    orientation="h",
                    text=[f"{v:+.3f}" for v in df_tor["Correlacao"]],
                    textposition="outside",
                    marker_color=df_tor["Cor"].tolist(),
                    customdata=df_tor[["Tipo", "Impacto"]].values.tolist(),
                    hovertemplate="<b>%{y}</b><br>Correlação (r): %{x:+.3f}<br>Dimensão: %{customdata[0]}<br>Classificação: %{customdata[1]}<extra></extra>",
                    cliponaxis=False
                ))
                fig_tor.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    xaxis=dict(title="Correlação Linear de Pearson (r)", range=[-0.50, 0.80]),
                    yaxis=dict(title=None),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_tor, use_container_width=True)
                
            with col_t2:
                st.markdown("##### Autocorrelação Serial (Ciclos Semanais)")
                st.caption("Picos periódicos nos lags 5, 10 e 15 provam a forte sazonalidade semanal de 5 dias úteis.")
                
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
                df_autocorr["Cor"] = df_autocorr["Destaque"].apply(lambda d: "#008744" if "Pico" in d else "#94a3b8")
                
                fig_auto = go.Figure(go.Bar(
                    x=[f"Lag {l}" for l in df_autocorr["Lag"]],
                    y=df_autocorr["Autocorrelacao"].tolist(),
                    text=[f"{v:.2f}" for v in df_autocorr["Autocorrelacao"]],
                    textposition="outside",
                    marker_color=df_autocorr["Cor"].tolist(),
                    customdata=df_autocorr["Destaque"].tolist(),
                    hovertemplate="<b>%{x}</b><br>Autocorrelação: %{y:.2f}<br>%{customdata}<extra></extra>",
                    cliponaxis=False
                ))
                fig_auto.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    xaxis=dict(title="Defasagem Temporal (Dias Úteis)"),
                    yaxis=dict(title="Autocorrelação", range=[min(df_autocorr["Autocorrelacao"])*1.2 if min(df_autocorr["Autocorrelacao"]) < 0 else 0, max(df_autocorr["Autocorrelacao"])*1.25]),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_auto, use_container_width=True)

            with st.expander("🖼️ Ver Figuras Acadêmicas da Dissertação (Figuras 10, 11 e 12)", expanded=False):
                col_f_e1, col_f_e2, col_f_e3 = st.columns(3)
                with col_f_e1:
                    p10 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_10_matriz_correlacao_multivariada.png")
                    if os.path.exists(p10):
                        st.image(p10, caption="Figura 10: Matriz de Correlação Multivariada", use_container_width=True)
                with col_f_e2:
                    p11 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_11_autocorrelacao_temporal.png")
                    if os.path.exists(p11):
                        st.image(p11, caption="Figura 11: Autocorrelação Serial (Lags 1 a 20)", use_container_width=True)
                with col_f_e3:
                    p12 = os.path.join(ROOT_DIR, "resultados", "figuras", "eda", "figura_12_ranking_impacto_tornado.png")
                    if os.path.exists(p12):
                        st.image(p12, caption="Figura 12: Tornado Chart de Poder Preditivo", use_container_width=True)

elif pagina_ativa == "Previsões":
    st.markdown('<div class="story-badge">ETAPA 2 DE 6 • ORDEM DE PREPARO SEMANAL (t+1)</div>', unsafe_allow_html=True)
    st.markdown("### Previsões e Ordem de Preparo Semanal")
    st.markdown('<p class="story-subtitle">Toda sexta-feira: consulte a recomendação diária calibrada para a próxima semana útil, embasada no modelo híbrido, que combina o ExtraTrees e o quantitativo informado pela nutricionista.</p>', unsafe_allow_html=True)
    
    renderizar_painel_explicativo(
        objetivo_rotina=(
            "**Saber a quantidade exata de refeições a cozinhar em cada dia da próxima semana útil.** "
            "Elimina o 'chutômetro' da equipe de cozinha e protege contra a falta de comida nos horários de pico."
        ),
        como_foi_feito_ciencia=(
            "**Combinação Inteligente (Modelo Híbrido 50/50 + Margem de Segurança):** "
            "O sistema junta a experiência prática da nutricionista com modelos preditivos de inteligência artificial e aprendizado de máquina. "
            "Para reduzir o risco de falta, adiciona uma margem calculada a partir das faltas observadas de fevereiro a agosto de 2026."
        ),
        como_usar_passos=(
            "1. **Toda sexta-feira:** Veja os cartões abaixo com a 'Ordem de Preparo' para cada dia da semana.\n"
            "2. **Clique em 'Baixar Grade de Produção'** para salvar a planilha e encaminhar para a equipe da cozinha terceirizada.\n"
            "3. **Use o Simulador abaixo** se quiser testar o que acontece ao trocar um prato ou se começar a chover forte."
        )
    )
    
    # Configuração de Modelos (Expander Opcional para Auditoria)
    with st.expander("Ajuste Fino de Algoritmos e Margens de Segurança (Opcional)", expanded=False):
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            cenario_t1 = st.selectbox("Cenário de Dados (Ablation Study):", list(CENARIOS_DICT.keys()), index=list(CENARIOS_DICT.keys()).index(CENARIO_PADRAO), key="cfg_cen_t1")
            st.caption(f"{CENARIOS_DICT[cenario_t1]['descricao']}")
        with col_c2:
            modelo_t1 = st.selectbox("Algoritmo Ativo:", list(MODELOS_RANKING.keys()), index=0, key="cfg_mod_t1")
        with col_c3:
            margem_t1 = st.selectbox("Margem de Segurança:", list(MARGENS_INFO.keys()), index=0, key="cfg_mar_t1")

    df_t1, df_feat_t1 = executar_pipeline_previsao(
        horizonte_alvo="Semana_t1",
        cenario_escolhido=cenario_t1,
        modelo_escolhido=modelo_t1,
        tipo_margem=margem_t1
    )

    if not df_t1.empty:
        # PAINEL DE TRANSPARÊNCIA E PROVENIÊNCIA DE DADOS (SEMANA T+1)
        renderizar_painel_transparencia_dados(
            df_t1,
            titulo_painel="Diagnóstico de Insumos e Transparência dos Dados — Semana t+1 (Ordem de Preparo)",
            horizonte_rotulo="Semana t+1"
        )

        total_semana_t1 = int(df_t1["recomendacao_final"].sum())
        df_letivos_t1 = df_t1[df_t1["is_dia_letivo"] == True]
        media_dia_t1 = int(df_letivos_t1["recomendacao_final"].mean()) if not df_letivos_t1.empty else 0
        dias_letivos_count = len(df_letivos_t1)
        dias_nao_letivos_count = len(df_t1) - dias_letivos_count

        # INDICADORES DA SEMANA T+1
        c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
        with c_kpi1:
            st.metric("Volume Semanal de Preparo", f"{total_semana_t1:,} refeições".replace(",", "."))
        with c_kpi2:
            st.metric("Média Diária em Dias Letivos", f"{media_dia_t1} ref/dia", delta=f"{dias_letivos_count} dias operacionais")
        with c_kpi3:
            st.metric("Calendário Escolar", f"{dias_letivos_count} dias com atendimento", delta=f"{dias_nao_letivos_count} feriado/fechado" if dias_nao_letivos_count > 0 else "Semana 100% Letiva")
        with c_kpi4:
            margem_media = int(df_letivos_t1["margem_aplicada"].mean()) if not df_letivos_t1.empty else 0
            st.metric("Margem Protetiva Embutida", f"+{margem_media} ref/dia", delta=f"Política: {margem_t1.split('(')[0].strip()}")

        st.markdown("---")
        # CARTOES DIARIOS HORIZONTAIS DA SEMANA (GRADE DA COZINHA)
        st.markdown("##### Sugestão de Preparo para os Próximos 5 Dias Úteis (Calendário Acadêmico)")
        
        cards_items = []
        for _, row_dia in df_t1.iterrows():
            d_nome = html.escape(str(row_dia["dia_semana"]).upper())
            d_data = html.escape(str(row_dia["data"]))
            d_prato = html.escape(str(row_dia["prato_principal"]))
            d_ordem = int(row_dia["recomendacao_final"])
            d_base = int(row_dia["pred_base_combinada"])
            d_marg = int(row_dia["margem_aplicada"])
            is_let = bool(row_dia.get("is_dia_letivo", True))
            status_card = str(row_dia.get("status_cardapio", ""))
            temp_dia = float(row_dia.get("temp_max_c", 25.0))
            chuva_dia = float(row_dia.get("chuva_mm", 0.0))
            
            if not is_let:
                card_html = (
                    f'<div class="day-prod-card" style="opacity: 0.9; border-top: 4px solid #94a3b8 !important; border-color: #cbd5e1 !important; background: #f8fafc;">'
                    f'<div class="day-prod-date">{d_nome} ({d_data[:5]})</div>'
                    f'<div class="day-prod-dish" style="color: #64748b; font-weight: 600;">{d_prato}</div>'
                    f'<div class="day-prod-number" style="color: #94a3b8;">0</div>'
                    f'<div class="day-prod-badge" style="background: #fee2e2; color: #991b1b; border: 1px solid #fecaca;">⛔ Feriado / Fechado (0 ref)</div>'
                    f'<div class="day-prod-sub" style="color: #64748b; margin-top: 4px;">Não Letivo • Refeitório Fechado</div>'
                    f'</div>'
                )
            else:
                card_menu_pill = (
                    f'<span style="font-size: 10.5px; background: #dcfce7; color: #166534; padding: 2px 6px; border-radius: 4px; font-weight: 600;">✅ Cardápio Oficial</span>'
                    if "Oficial" in status_card
                    else f'<span style="font-size: 10.5px; background: #fef3c7; color: #92400e; padding: 2px 6px; border-radius: 4px; font-weight: 600;">⚠️ Cardápio Estimado (Neutro)</span>'
                )
                clima_tag = f'<div style="font-size: 11px; color: #64748b; margin-top: 4px;">🌦️ {temp_dia:.1f}°C | {chuva_dia:.1f} mm</div>'
                card_html = (
                    f'<div class="day-prod-card">'
                    f'<div class="day-prod-date">{d_nome} ({d_data[:5]})</div>'
                    f'<div class="day-prod-dish">{d_prato}</div>'
                    f'<div class="day-prod-number">{d_ordem}</div>'
                    f'<div class="day-prod-badge">Sugestão de Preparo</div>'
                    f'<div class="day-prod-sub">Base: {d_base} | Margem: +{d_marg}</div>'
                    f'<div style="margin-top: 8px;">{card_menu_pill}</div>'
                    f'{clima_tag}'
                    f'</div>'
                )
            cards_items.append(card_html)
            
        full_grid_html = f'<div class="day-prod-grid">{"".join(cards_items)}</div>'
        st.markdown(full_grid_html, unsafe_allow_html=True)

        # TABELA OPERACIONAL COMPLETA (APOIO A DECISAO)
        st.markdown("##### Tabela Operacional de Produção — Apoio à Decisão da Nutricionista")
        df_exibir = pd.DataFrame()
        df_exibir["Data"] = df_t1["data"]
        df_exibir["Dia"] = df_t1["dia_semana"]
        df_exibir["Calendário"] = df_t1.apply(
            lambda r: f"⛔ {r['evento_tipo']}" if not r.get("is_dia_letivo", True) else "✅ Letivo Regular",
            axis=1
        )
        df_exibir["Prato Principal"] = df_t1["prato_principal"]
        df_exibir["Status Cardápio"] = df_t1["status_cardapio"]
        df_exibir["Fonte Clima"] = df_t1["status_clima"]
        df_exibir["Previsão do Tempo"] = df_t1.apply(
            lambda r: f"{r['temp_max_c']:.1f}°C / {r['chuva_mm']:.1f} mm",
            axis=1
        )
        
        def formatar_nutri_t1(row):
            if not row.get("is_dia_letivo", True):
                return "Não se aplica (Feriado)"
            val = row.get("estimativa_nutri")
            if pd.isna(val) or val is None or val == 0:
                return "A definir (Autônomo)"
            return f"{int(round(val))} ref"
            
        df_exibir["Quantitativo Nutrição (Meta)"] = df_t1.apply(formatar_nutri_t1, axis=1)
        df_exibir["Previsão Base (IA)"] = df_t1["ia_pura"].astype(int).astype(str) + " ref"
        df_exibir["Margem Segura"] = "+" + df_t1["margem_aplicada"].astype(int).astype(str) + " ref"
        
        def formatar_sugestao_t1(row):
            if not row.get("is_dia_letivo", True):
                return "0 refeições (Fechado)"
            return f"{int(row.get('recomendacao_final', 0))} refeições"
            
        df_exibir["Sugestão de Preparo (Final)"] = df_t1.apply(formatar_sugestao_t1, axis=1)
        
        def formatar_faixa_t1(row):
            if not row.get("is_dia_letivo", True):
                return "0 ref (Fechado)"
            base = row.get("pred_base_combinada", 0)
            rec = row.get("recomendacao_final", 0)
            f_min = max(0, int(base - 8))
            f_max = int(rec + 8)
            return f"{f_min} a {f_max} ref"
            
        df_exibir["Faixa Sugerida (Mín - Máx)"] = df_t1.apply(formatar_faixa_t1, axis=1)
        
        def formatar_alertas(al_txt):
            if not al_txt or pd.isna(al_txt):
                return "Operação regular"
            return str(al_txt).replace("\ufe0f", "").strip()
            
        df_exibir["Alertas"] = df_t1["alertas"].apply(formatar_alertas)
        st.dataframe(df_exibir, use_container_width=True, hide_index=True)

        # Exportação CSV
        col_exp1, col_exp2 = st.columns([1.2, 3])
        with col_exp1:
            csv_grade = df_exibir.to_csv(index=False, sep=";").encode("utf-8-sig")
            st.download_button(
                label="Baixar Grade de Produção (Planilha CSV)",
                data=csv_grade,
                file_name=f"ordem_preparo_refeicoes_{date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.warning("Não foram identificadas datas letivas para o horizonte selecionado. Verifique o calendário operacional do campus.")

    st.markdown("---")

    st.markdown("---")
    with st.container(border=True):
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px; padding: 6px;">
            <div>
                <div style="font-weight: 700; font-size: 15.5px; color: var(--text-primary); margin-bottom: 4px;">
                     Quer simular datas específicas, modelos ou cenários alternativos?
                </div>
                <div style="font-size: 13px; color: var(--text-muted); line-height: 1.5;">
                    O <b>Simulador de Cenários</b> possui uma página dedicada na barra superior! Nele você pode escolher qualquer data (passada ou futura), comparar os modelos da pesquisa (Modelo Híbrido, ExtraTrees e planejamento manual), ajustar margens de proteção alimentar e verificar a influência de cardápio e clima em tempo real.
                </div>
            </div>
            <div>
                <span class="badge-status" style="cursor: default; font-size: 13px; padding: 6px 14px;"> Acesse na Barra Superior: <b>🔬 Simulador de Cenários</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# PÁGINA 3: SIMULADOR DE CENÁRIOS ("E SE...?")
# ==============================================================================
elif pagina_ativa == "Simulador de Cenários":
    st.markdown('<div class="story-badge">ETAPA 3 DE 6 • LABORATÓRIO DE SIMULAÇÕES ("E SE...?")</div>', unsafe_allow_html=True)
    st.markdown("### Simulador de Cenários e Decisão Operacional")
    st.markdown('<p class="story-subtitle">Simule uma data como se a decisão fosse tomada na sexta de corte. Compare os modelos da pesquisa, cenários de dados e margens de proteção alimentar.</p>', unsafe_allow_html=True)
    
    renderizar_painel_explicativo(
        objetivo_rotina=(
            "**Realizar simulações controladas para qualquer data do calendário letivo.** "
            "Permite à nutricionista e à gestão testar o impacto de substituições de proteínas no cardápio, "
            "cenários de tempestade e diferentes margens de segurança antes de autorizar a ordem de preparo."
        ),
        como_foi_feito_ciencia=(
"**Simulação na sexta de corte.** "
            "O artefato usa somente o consumo anterior à sexta da emissão e a previsão de clima arquivada, nunca o clima observado. "
            "Datas até 31/08/2026 são refeitas em walk-forward com treino até aquela sexta. As demais usam o modelo congelado em 31/08/2026."
        ),
        como_usar_passos=(
            "1. **Escolha a data no calendário** e defina o objetivo (Consumo diário t+1 ou Compras quinzenais t+2).\n"
            "2. **Selecione o Cenário e o Modelo** da pesquisa que deseja avaliar.\n"
            "3. **Confira a validação de dados da base**: caso falte cardápio, clima ou quantitativo institucional para a data, o sistema emite alerta e permite informar valores hipotéticos.\n"
            "4. **Analise a recomendação gerada**, alertas operacionais contextuais e a auditoria retroativa (backtest) caso a data já tenha ocorrido."
        )
    )

    df_fato_completo = carregar_fato_diaria_completa()
    df_planejamento_base = carregar_planejamento_nutricionista()

    col_ctrl, col_valid = st.columns([1.15, 1.25])

    with col_ctrl:
        with st.container(border=True):
            st.markdown("##### 1. Parâmetros da Simulação")
            st.caption("Defina a data-alvo, objetivo operacional e os modelos treinados da pesquisa.")
            
            # Data padrão: próximo dia útil
            proximos_uteis = obter_proximos_dias_uteis(num_dias=5)
            if proximos_uteis:
                d_cand = proximos_uteis[0]
                data_padrao = d_cand.date() if (hasattr(d_cand, "date") and callable(d_cand.date)) else d_cand
            else:
                data_padrao = date.today() + timedelta(days=1)
            
            data_sim = st.date_input(
                "Selecione a Data para Simular:",
                value=data_padrao,
                min_value=date(2023, 1, 1),
                max_value=date(2026, 12, 31),
                key="sim_data_input"
            )
            
            objetivo_opcoes = [
                "Consumo Diário / Ordem de Preparo (t+1)",
                "Compras & Suprimentos (t+2 Quinzenal)"
            ]
            objetivo_sel = st.radio(
                "Objetivo da Previsão:",
                objetivo_opcoes,
                index=0,
                key="sim_obj_radio"
            )
            obj_chave = "consumo" if "Consumo" in objetivo_sel else "compras"
            
            # Em Compras (t+2) não existe quantitativo institucional
            if obj_chave == "compras":
                opcoes_cenarios = [c for c in CENARIOS_DICT.keys() if "Cenário 4" not in c]
                opcoes_modelos = [m for m in MODELOS_RANKING.keys() if "Híbrido" not in m and "Manual" not in m]
            else:
                opcoes_cenarios = list(CENARIOS_DICT.keys())
                opcoes_modelos = list(MODELOS_RANKING.keys())
                
            cenario_sel = st.selectbox(
                "Cenário de Dados (Ablation Study):",
                opcoes_cenarios,
                index=opcoes_cenarios.index(CENARIO_PADRAO),
                key="sim_cenario_select"
            )
            st.caption(f"ℹ️ {CENARIOS_DICT[cenario_sel]['descricao']}")
            
            modelo_sel = st.selectbox(
                "Modelo ou Algoritmo Preditivo:",
                opcoes_modelos,
                index=0,
                key="sim_modelo_select"
            )
            st.caption(f"💡 **Como funciona:** {MODELOS_RANKING[modelo_sel]['detalhe']}")
            
            margens_opcoes = list(MARGENS_INFO.keys()) + ["Margem Personalizada (Manual)"]
            margem_sel = st.selectbox(
                "Política de Margem de Segurança:",
                margens_opcoes,
                index=0,
                key="sim_margem_select"
            )
            if margem_sel == "Margem Personalizada (Manual)":
                margem_custom_val = st.number_input(
                    "Refeições Extras de Margem (+ref):",
                    min_value=0,
                    max_value=60,
                    value=12,
                    step=1,
                    key="sim_margem_custom_num"
                )
            else:
                margem_custom_val = 0.0
                st.caption(f"{MARGENS_INFO[margem_sel]['descricao']} (Cobertura: {MARGENS_INFO[margem_sel]['cobertura']})")

    with col_valid:
        with st.container(border=True):
            st.markdown("##### 2. Validação dos Dados na Base para a Data")
            dia_sem_num = data_sim.weekday()
            dia_sem_rotulo = DIAS_SEMANA_PT.get(dia_sem_num, "Dia")
            st.caption(f"Verificação de integridade para **{data_sim.strftime('%d/%m/%Y')}** ({dia_sem_rotulo}).")
            
            dt_alvo_ts = pd.to_datetime(data_sim)
            data_iso_str = dt_alvo_ts.strftime("%Y-%m-%d")
            
            row_fato = df_fato_completo[df_fato_completo["data_dt"] == dt_alvo_ts]
            row_plan = df_planejamento_base[df_planejamento_base["data_iso"] == data_iso_str]
            
            is_letivo_sim = True
            evento_sim = "Letivo Regular"
            if not row_fato.empty:
                is_letivo_sim = bool(row_fato.iloc[0].get("is_dia_letivo", True))
                evento_sim = str(row_fato.iloc[0].get("evento_tipo", "Letivo Regular"))
                status_leg = str(row_fato.iloc[0].get("status_legal", "")).lower().strip()
                if status_leg == "nao_letivo" or evento_sim.lower() in ["feriado", "recesso", "ferias", "sem_atividade_letiva", "final_de_semana", "ponto_facultativo"]:
                    is_letivo_sim = False
            elif data_sim.weekday() >= 5:
                is_letivo_sim = False
                evento_sim = "final_de_semana"
            
            chave_sim = data_iso_str[5:]
            nome_fer = FERIADOS_FIXOS_BR.get(chave_sim)
            if not is_letivo_sim:
                rotulo_fechado = f"Feriado Nacional ({nome_fer})" if nome_fer else evento_sim.replace('_', ' ').title()
                st.error(f"⛔ **Data Não Letiva no Calendário Acadêmico:** {data_sim.strftime('%d/%m/%Y')} está classificada como **{rotulo_fechado}**. O restaurante não opera refeições regulares nesta data (sugestão = 0 refeições).")
            
            cen_cfg = CENARIOS_DICT[cenario_sel]
            usar_cardapio = cen_cfg["usar_cardapio"]
            usar_clima = cen_cfg["usar_clima"]
            usar_nutri = (cen_cfg.get("usar_nutri", False) or "Híbrido" in modelo_sel or "Manual" in modelo_sel) and (obj_chave == "consumo")
            
            # --- Validação A: Cardápio ---
            st.markdown("######  Cardápio (Prato Principal)")
            tem_cardapio = False
            prato_existente = ""
            prot_existente = "frango"
            prep_existente = "assado"
            
            if not row_plan.empty and pd.notna(row_plan.iloc[0].get("prato_principal")) and str(row_plan.iloc[0].get("prato_principal")).strip() != "":
                tem_cardapio = True
                prato_existente = str(row_plan.iloc[0].get("prato_principal"))
                prot_existente = str(row_plan.iloc[0].get("proteina_tipo", "frango")).lower()
                prep_existente = str(row_plan.iloc[0].get("preparo_tipo", "assado"))
            elif not row_fato.empty and pd.notna(row_fato.iloc[0].get("prato_principal")) and str(row_fato.iloc[0].get("prato_principal")).strip() != "":
                tem_cardapio = True
                prato_existente = str(row_fato.iloc[0].get("prato_principal"))
                prot_existente = str(row_fato.iloc[0].get("proteina_tipo", "frango")).lower()
                prep_existente = str(row_fato.iloc[0].get("preparo_tipo", "assado"))
                
            prato_custom_override = None
            prot_custom_override = None
            prep_custom_override = None
            
            if usar_cardapio:
                if tem_cardapio:
                    st.success(f"**Cadastrado:** {prato_existente} ({prot_existente.title()} / {prep_existente})")
                    simular_outro_prato = st.checkbox("Substituir por prato hipotético para teste?", value=False, key="sim_chk_prato")
                    if simular_outro_prato:
                        prot_custom_override = st.selectbox(
                            "Proteína Hipotética:",
                            ["bovina", "frango", "suina", "peixe", "ovo"],
                            format_func=lambda x: {"bovina": "Carne Bovina (Alta Procura)", "frango": " Frango / Aves (Boa Procura)", "suina": " Carne Suína (Média Procura)", "peixe": " Peixe / Pescados (Menor Procura)", "ovo": " Ovo / Opção Vegetariana"}.get(x, x),
                            key="sim_prot_sub"
                        )
                        prato_custom_override = st.text_input("Nome do Prato Hipotético:", value=f"Especial ({prot_custom_override.title()})", key="sim_nome_prato_sub")
                else:
                    st.warning("**Cardápio não cadastrado na base para esta data.**")
                    st.caption("Cadastre oficialmente em **✍️ Cadastrar Dados** ou selecione a proteína hipotética abaixo para testar o efeito no modelo:")
                    prot_custom_override = st.selectbox(
                        "Proteína Hipotética para a Simulação:",
                        ["bovina", "frango", "suina", "peixe", "ovo"],
                        index=0,
                        format_func=lambda x: {"bovina": "Carne Bovina (Alta Procura)", "frango": " Frango / Aves (Boa Procura)", "suina": " Carne Suína (Média Procura)", "peixe": " Peixe / Pescados (Menor Procura)", "ovo": " Ovo / Opção Vegetariana"}.get(x, x),
                        key="sim_prot_empty"
                    )
                    prato_custom_override = f"Prato Simulado ({prot_custom_override.title()})"
            else:
                st.info("ℹ️ O cenário escolhido não utiliza cardápio.")
                
            # --- Validação B: Clima ---
            st.markdown("###### Condições Climáticas")
            c_prev, t_prev, origem_clima_rotulo = obter_clima_emissao(data_sim, "t1" if obj_chave == "consumo" else "t2")
            tem_clima = c_prev is not None and t_prev is not None
            chuva_existente = float(c_prev) if tem_clima else 0.0
            temp_existente = float(t_prev) if tem_clima else 25.0

            chuva_custom_override = None
            temp_custom_override = None
            
            if usar_clima:
                if tem_clima:
                    st.success(f"**Clima:** {chuva_existente:.1f} mm chuva | {temp_existente:.1f}°C máx ({origem_clima_rotulo})")
                    simular_outro_clima = st.checkbox("Simular condição climática diferente?", value=False, key="sim_chk_clima")
                    if simular_outro_clima:
                        chuva_custom_override = st.slider("Chuva Hipotética (mm):", min_value=0.0, max_value=40.0, value=chuva_existente, step=1.0, key="sim_chuva_slider_sub")
                        temp_custom_override = st.slider("Temperatura Máx (°C):", min_value=12.0, max_value=38.0, value=temp_existente, step=0.5, key="sim_temp_slider_sub")
                else:
                    st.warning("**Previsão de clima não arquivada para esta data.**")
                    st.caption("Ajuste a meteorologia estimada para que os algoritmos calculem a sensibilidade climática:")
                    chuva_custom_override = st.slider("Precipitação Prevista (mm):", min_value=0.0, max_value=40.0, value=0.0, step=1.0, key="sim_chuva_slider_empty")
                    temp_custom_override = st.slider("Temperatura Máxima (°C):", min_value=12.0, max_value=38.0, value=25.0, step=0.5, key="sim_temp_slider_empty")
            else:
                st.info("ℹ️ O cenário escolhido não utiliza variáveis meteorológicas.")
                
            # --- Validação C: Nutrição ---
            nutri_custom_override = None
            if usar_nutri:
                st.markdown("######  Quantitativo Institucional da Nutrição")
                tem_nutri = False
                nutri_existente = None
                if not row_plan.empty and pd.notna(row_plan.iloc[0].get("planejado_nutri")) and float(row_plan.iloc[0].get("planejado_nutri", 0)) > 0:
                    tem_nutri = True
                    nutri_existente = float(row_plan.iloc[0]["planejado_nutri"])
                elif not row_fato.empty and pd.notna(row_fato.iloc[0].get("planejado_nutri")) and float(row_fato.iloc[0].get("planejado_nutri", 0)) > 0:
                    tem_nutri = True
                    nutri_existente = float(row_fato.iloc[0]["planejado_nutri"])
                    
                if tem_nutri:
                    st.success(f"**Meta na Base:** {int(round(nutri_existente))} refeições")
                    simular_outro_nutri = st.checkbox("Simular outro quantitativo institucional?", value=False, key="sim_chk_nutri")
                    if simular_outro_nutri:
                        nutri_custom_override = st.number_input("Novo Quantitativo:", min_value=50, max_value=300, value=int(round(nutri_existente)), step=1, key="sim_nutri_num_sub")
                else:
                    st.warning("**Meta da Nutrição não cadastrada para esta data.**")
                    st.caption("Informe o valor para compor o modelo híbrido. Sem ele, o resultado usa apenas o ExtraTrees.")
                    valor_nutri_sim = st.number_input("Quantitativo Institucional da Simulação (0 = não informado):", min_value=0, max_value=300, value=0, step=1, key="sim_nutri_num_empty")
                    nutri_custom_override = valor_nutri_sim if valor_nutri_sim > 0 else None

    # Executar a simulação na sexta de corte da data escolhida
    resultado_sim = simular_previsao_data(
        data_alvo=data_sim,
        objetivo=obj_chave,
        cenario_escolhido=cenario_sel,
        modelo_escolhido=modelo_sel,
        tipo_margem=margem_sel,
        margem_customizada=margem_custom_val,
        proteina_custom=prot_custom_override,
        prato_custom=prato_custom_override,
        preparo_custom=prep_custom_override,
        chuva_custom=chuva_custom_override,
        temp_custom=temp_custom_override,
        nutri_custom=nutri_custom_override
    )

    st.markdown("---")
    st.markdown("#### Resultado da Simulação")
    st.caption(f"Sexta de corte: {resultado_sim['origem'].strftime('%d/%m/%Y')}. Modelo usado: {resultado_sim['regime_modelo']}.")

    # Indicadores Principais
    c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
    with c_kpi1:
        rotulo_prod = "Ordem de Preparo (t+1)" if obj_chave == "consumo" else "Volume para Compras (t+2)"
        if not resultado_sim["is_dia_letivo"]:
            st.metric(
                rotulo_prod,
                "0 refeições",
                delta="Feriado / Sem Atendimento",
                delta_color="off"
            )
        else:
            delta_ref = resultado_sim["pred_final"] - 180
            st.metric(
                rotulo_prod,
                f"{resultado_sim['pred_final']} refeições",
                delta=f"{delta_ref:+d} vs média regular (180)"
            )
    with c_kpi2:
        if not resultado_sim["is_dia_letivo"]:
            st.metric("Previsão Bruta (IA)", "0 ref", delta="Não aplicável (Fechado)", delta_color="off")
        else:
            st.metric("Previsão Bruta (IA)", f"{resultado_sim['pred_ia']} ref", delta="Sem margem extra")
    with c_kpi3:
        if not resultado_sim["is_dia_letivo"]:
            st.metric("Margem de Segurança", "+0 ref", delta="Não aplicável (Fechado)", delta_color="off")
        else:
            st.metric("Margem de Segurança", f"+{int(round(resultado_sim['margem_val']))} ref", delta=f"Política: {margem_sel.split('(')[0].strip()}")
    with c_kpi4:
        st.metric(
            "Histórico até a Sexta de Corte",
            f"{resultado_sim['dias_historico_utilizados']} dias",
            delta=f"MM 5d: {resultado_sim['mm5']} ref"
        )

    # Backtest / Confronto se a data já ocorreu
    if resultado_sim["consumo_real"] is not None:
        real_val = int(round(resultado_sim["consumo_real"]))
        pred_val = resultado_sim["pred_final"]
        erro_abs = abs(pred_val - real_val)
        saldo_dia = pred_val - real_val
        
        with st.container(border=True):
            st.markdown("#####  Auditoria Retroativa (Backtest da Data Real)")
            st.caption(f"Esta data já ocorreu no histórico e possui consumo real auditado: **{real_val} refeições servidas**.")
            
            c_bt1, c_bt2, c_bt3 = st.columns(3)
            with c_bt1:
                st.metric("Consumo Real dos Alunos", f"{real_val} ref")
            with c_bt2:
                st.metric("Recomendação do Modelo", f"{pred_val} ref", delta=f"Erro absoluto: {erro_abs} ref")
            with c_bt3:
                if saldo_dia >= 0:
                    st.metric("Balanço Protetivo", f"+{saldo_dia} ref", delta="Sobra controlada / Desabastecimento evitado")
                else:
                    st.metric("Balanço Protetivo", f"{saldo_dia} ref", delta="Déficit potencial")
                    
            if saldo_dia >= 0:
                st.success(f"**Proteção Alimentar Assegurada:** A recomendação com margem ({pred_val} ref) atendeu plenamente aos estudantes ({real_val} ref), gerando folga segura de {saldo_dia} refeições sem risco de desabastecimento.")
            else:
                st.warning(f"**Atenção:** A estimativa calculada ficou {abs(saldo_dia)} refeições abaixo da demanda real verificada. Avalie a política de margem para datas com esta volatilidade.")

    # Alertas Operacionais Contextuais
    if resultado_sim["alertas"]:
        st.markdown("##### Diagnóstico e Alertas Operacionais:")
        for alerta in resultado_sim["alertas"]:
            if "️" in alerta or "Alerta" in alerta:
                st.error(alerta)
            elif "️" in alerta or "️" in alerta or "Sexta-feira" in alerta or "Peixe" in alerta:
                st.warning(alerta)
            elif "" in alerta:
                st.success(alerta)
            else:
                st.info(alerta)

    # Detalhes Científicos e Vetor de Features
    with st.expander(" Detalhes Científicos e Atributos Utilizados"):
        st.markdown(f"""
        **Corte temporal:** as médias móveis e defasagens de consumo usam apenas dias anteriores à sexta de corte ({resultado_sim['origem'].strftime('%d/%m/%Y')}).
        Dias com registro de atípico e dias com consumo muito baixo não entram nessas médias.
        """)
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("**Defasagens Temporais Apuradas no Histórico:**")
            st.write(f"- Média Móvel de 5 dias úteis (`mm5`): **{resultado_sim['mm5']} ref**")
            st.write(f"- Consumo do mesmo dia na semana anterior (`lag5`): **{resultado_sim['lag5']} ref**")
            st.write(f"- Consumo de duas semanas antes (`lag10`): **{resultado_sim['lag10']} ref**")
            st.write(f"- Total de dias letivos anteriores auditados: **{resultado_sim['dias_historico_utilizados']} dias**")
        with col_f2:
            st.markdown("**Entradas Consolidadas para os Modelos:**")
            st.write(f"- Proteína considerada: **{resultado_sim['prot_utilizada'].title()}** (Prato: {resultado_sim['prato_utilizado']})")
            st.write(f"- Precipitação prevista: **{resultado_sim['chuva_utilizada']:.1f} mm**")
            st.write(f"- Temperatura máxima prevista: **{resultado_sim['temp_utilizada']:.1f}°C**")
            if resultado_sim.get("nutri_utilizado") is not None and usar_nutri:
                st.write(f"- Quantitativo institucional (Nutri): **{resultado_sim['nutri_utilizado']:.1f} ref**")

# ==============================================================================
# PÁGINA 4: COMPRAS & SUPRIMENTOS (JANELA LONGA t+2: 14 DIAS)
# ==============================================================================
elif pagina_ativa == "Compras & Suprimentos":
    st.markdown('<div class="story-badge">ETAPA 4 DE 6 • PLANEJAMENTO DE SUPRIMENTOS (t+2)</div>', unsafe_allow_html=True)
    st.markdown("###  Planejamento de Compras e Estoque — Janela Longa (10 a 14 dias)")
    st.markdown('<p class="story-subtitle">A cada 15 dias: dimensione pedidos antecipados de carnes congeladas e mantimentos de despensa, sem depender de quantitativo humano.</p>', unsafe_allow_html=True)
    
    renderizar_painel_explicativo(
        objetivo_rotina=(
            "**Planejar a compra antecipada de mantimentos de despensa e alimentos estocáveis (t+2).** "
            "Ajuda a equipe de suprimentos a dimensionar pedidos quinzenais de carnes congeladas, arroz, feijão e faturamento contratual."
        ),
        como_foi_feito_ciencia=(
"**Modelo de Longo Prazo (ExtraTrees, Cenário 2: Consumo + Calendário + Cardápio):** "
            "A 14 dias de distância, ainda não existe quantitativo institucional. "
            "Por isso, as compras usam apenas o histórico recente, o calendário escolar e o cardápio já preparado."
        ),
        como_usar_passos=(
            "1. **Consulte a cada 15 dias** o volume quinzenal total estimado.\n"
            "2. **Utilize o botão de download** para exportar os números para o setor de compras ou fornecedor.\n"
            "3. **Na sexta-feira anterior à produção**, a ordem de preparo diária é refinada na tela de Previsões."
        )
    )
    
    st.info(
        "**Apoio à Decisão e Soberania Técnica:** "
        "Os quantitativos calculados abaixo representam **sugestões de preparo e dimensionamento antecipado de insumos** gerados pelo modelo da pesquisa com margem de segurança por quantil. "
        "A decisão operacional final de compra, faturamento contratual e preparo pertence integralmente à Nutricionista Responsável Técnica e à Gestora do Restaurante."
    )
    
    with st.expander("Ajuste Fino de Modelo e Margens para Suprimentos (Opcional)", expanded=False):
        st.caption("Modelo das compras: ExtraTrees congelado em 31/08/2026, sem quantitativo institucional.")
        margem_sup = st.selectbox(
            "Margem Protetiva:",
            list(MARGENS_INFO.keys()),
            index=0,
            key="cfg_mar_sup"
        )

    cenario_sup = CENARIO_PADRAO
    modelo_nome_exec = MODELO_IA

    df_t2, df_feat_t2 = executar_pipeline_previsao(
        horizonte_alvo="Pos_Semana_t2",
        cenario_escolhido=cenario_sup,
        modelo_escolhido=modelo_nome_exec,
        tipo_margem=margem_sup
    )
    
    if not df_t2.empty:
        # PAINEL DE TRANSPARÊNCIA E PROVENIÊNCIA DE DADOS (SEMANA T+2)
        renderizar_painel_transparencia_dados(
            df_t2,
            titulo_painel="Diagnóstico de Insumos e Transparência dos Dados — Semana t+2 (10 a 14 dias)",
            horizonte_rotulo="Semana t+2"
        )

        total_semana_t2 = int(df_t2["recomendacao_final"].sum())
        df_letivos_t2 = df_t2[df_t2["is_dia_letivo"] == True]
        media_dia_t2 = int(df_letivos_t2["recomendacao_final"].mean()) if not df_letivos_t2.empty else 0
        dias_letivos_t2_count = len(df_letivos_t2)
        dias_fechados_t2_count = len(df_t2) - dias_letivos_t2_count
        
        c_tot1, c_tot2, c_tot3, c_tot4 = st.columns(4)
        with c_tot1:
            st.metric("Volume Total Sugerido (Semana t+2)", f"{total_semana_t2:,} refeições".replace(",", "."))
        with c_tot2:
            st.metric("Média Diária em Dias Letivos", f"{media_dia_t2} ref/dia", delta=f"{dias_letivos_t2_count} dias operacionais")
        with c_tot3:
            st.metric("Calendário Escolar", f"{dias_letivos_t2_count} dias com atendimento", delta=f"{dias_fechados_t2_count} feriado/fechado" if dias_fechados_t2_count > 0 else "Semana 100% Letiva")
        with c_tot4:
            margem_med_t2 = int(df_letivos_t2["margem_aplicada"].mean()) if not df_letivos_t2.empty else 0
            st.metric("Margem Protetiva Embutida", f"+{margem_med_t2} ref/dia", delta="Margem por quantil")
            
        st.markdown("---")
        st.markdown("#### Grade Quinzenal de Compras e Sugestão de Preparo")
        
        df_exibir_t2 = pd.DataFrame()
        df_exibir_t2["Data"] = df_t2["data"]
        df_exibir_t2["Dia da Semana"] = df_t2["dia_semana"]
        df_exibir_t2["Calendário"] = df_t2.apply(
            lambda r: f"⛔ {r['evento_tipo']}" if not r.get("is_dia_letivo", True) else "✅ Letivo Regular",
            axis=1
        )
        df_exibir_t2["Cardápio / Prato Previsto"] = df_t2.apply(
            lambda r: r["prato_principal"] if not r.get("is_dia_letivo", True) else f"{r['prato_principal']} ({r['proteina_tipo']})",
            axis=1
        )
        df_exibir_t2["Origem do Cardápio"] = df_t2["status_cardapio"]
        df_exibir_t2["Fonte Meteorológica"] = df_t2["status_clima"]
        df_exibir_t2["Previsão Base (IA)"] = df_t2["ia_pura"].astype(int).astype(str) + " ref"
        df_exibir_t2["Margem de Segurança"] = "+" + df_t2["margem_aplicada"].astype(int).astype(str) + " ref"
        df_exibir_t2["Sugestão de Preparo (Final)"] = df_t2.apply(
            lambda r: "0 refeições (Fechado)" if not r.get("is_dia_letivo", True) else f"{int(r['recomendacao_final'])} refeições",
            axis=1
        )
        df_exibir_t2["Previsão do Tempo"] = df_t2["temp_max_c"].map(lambda x: f"{x:.1f}°C") + " / " + df_t2["chuva_mm"].map(lambda x: f"{x:.1f} mm chuva")
        df_exibir_t2["Alertas e Condições"] = df_t2["alertas"]
        
        st.dataframe(df_exibir_t2, use_container_width=True, hide_index=True)
        
        csv_t2 = df_exibir_t2.to_csv(index=False, sep=";").encode("utf-8-sig")
        st.download_button(
            label="Baixar Grade de Compras e Sugestão de Preparo (CSV)",
            data=csv_t2,
            file_name=f"planejamento_compras_sugestao_preparo_t2_{date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ==============================================================================
# PÁGINA 5: CADASTRAR DADOS (LANÇAMENTO E CONSULTA DE ROTINA)
# ==============================================================================
elif pagina_ativa == "Cadastrar Dados":
    st.markdown('<div class="story-badge">ETAPA 5 DE 6 • ALIMENTAÇÃO CONTÍNUA & ROTINA</div>', unsafe_allow_html=True)
    st.markdown("### Lançamento e Atualização de Informações do Refeitório")
    st.markdown('<p class="story-subtitle">Diariamente após o almoço: gaste 1 minuto para lançar o consumo verificado e manter os modelos sempre afinados com a realidade discente.</p>', unsafe_allow_html=True)
    
    renderizar_painel_explicativo(
        objetivo_rotina=(
            "**Registrar a quantidade real de refeições servidas e atualizar o cardápio planejado.** "
            "É essa rotina rápida de 1 minuto que mantém o histórico recente usado nas previsões sempre atualizado."
        ),
        como_foi_feito_ciencia=(
            "**Alimentação Contínua da Base Histórica (Camada Medallion):** "
            "Cada novo dado lançado é integrado com segurança à base analítica, atualizando as médias móveis de 5 e 10 dias e o histórico de refeições servidas."
        ),
        como_usar_passos=(
            "1. **Diariamente após as 14h:** Digite o número de refeições servidas naquele dia.\n"
            "2. **Na sexta-feira:** Cadastre os pratos previstos para a próxima semana para que as previsões fiquem precisas.\n"
            "3. **Proteção por Senha:** O lançamento de dados exige a senha operacional da equipe."
        )
    )
    
    st.info(
        "💡 **Ambiente Seguro de Demonstração e Simulação:**\n\n"
        "Este painel é o artefato computacional da pesquisa de mestrado. "
        "Sinta-se à vontade para preencher, simular ou alterar valores nesta tela: **as alterações têm finalidade exclusivamente demonstrativa e pedagógica**, "
        "servindo para ilustrar como os modelos preditivos recalculam as previsões em tempo real. "
        "Os dados oficiais da escola não são modificados de forma permanente no servidor da nuvem, garantindo total liberdade para a banca examinadora e operadores testarem o sistema."
    )
    
    if not st.session_state.get("autenticado", False):
        st.warning("**Página Bloqueada para Edição (Modo de Apenas Leitura)**")
        st.markdown("""
        Para proteger os registros operacionais da instituição pública de ensino contra alterações acidentais, 
        o lançamento de dados é reservado aos profissionais autorizados.
        
        * **Faz parte da equipe autorizada?** Desbloqueie com a senha operacional no formulário abaixo.
        * **É membro da Banca Examinadora ou Visitante?** Você pode consultar qualquer data cadastrada no histórico abaixo (modo leitura).
        """)
        
        with st.expander("Desbloquear Edição com Senha Operacional", expanded=True):
            with st.form("form_login_aba4"):
                col_l1, col_l2 = st.columns([2, 1])
                with col_l1:
                    perfil_rapido = st.selectbox("Selecione seu perfil:", ["Nutricionista / Gestora", "Pesquisador (Admin)"], key="perfil_aba4")
                    senha_rapida = st.text_input("Digite a senha de acesso:", type="password", key="senha_aba4")
                with col_l2:
                    st.write("")
                    st.write("")
                    st.write("")
                    btn_desbloquear = st.form_submit_button("Desbloquear Edição", use_container_width=True)
                    
                if btn_desbloquear:
                    if perfil_rapido == "Nutricionista / Gestora" and senha_confere(senha_rapida, SENHA_NUTRI):
                        st.session_state["autenticado"] = True
                        st.session_state["usuario_nome"] = "Nutricionista Responsável / Gestora"
                        st.session_state["usuario_perfil"] = "operador"
                        st.success("Acesso liberado com sucesso!")
                        st.rerun()
                    elif perfil_rapido == "Pesquisador (Admin)" and senha_confere(senha_rapida, SENHA_ADMIN):
                        st.session_state["autenticado"] = True
                        st.session_state["usuario_nome"] = "Michel Silva de Souza (Admin)"
                        st.session_state["usuario_perfil"] = "admin"
                        st.success("Acesso de administrador liberado!")
                        st.rerun()
                    else:
                        st.error("Senha incorreta. Tente novamente.")
                        
        st.markdown("---")
        st.markdown("##### Consulta de Dados Cadastrados (Somente Leitura)")
        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            data_sel = st.date_input("Consultar Data:", value=date.today(), key="data_picker_view")
        data_iso = data_sel.strftime("%Y-%m-%d")
        data_fmt = data_sel.strftime("%d/%m/%Y")
        dados_dia = consultar_dados_completos_data(data_iso)
        
        with col_d2:
            if dados_dia["encontrado_base"]:
                st.info(f"**{data_fmt} ({dados_dia['dia_semana']}) localizada no histórico.**")
            else:
                st.caption(f"{data_fmt} ainda não possui registros consolidados na base.")
                
        # Painel de Diagnóstico Visual (Somente Leitura)
        c_v1, c_v2, c_v3, c_v4 = st.columns(4)
        with c_v1:
            if dados_dia['consumo'] is not None:
                st.markdown(f"""
                <div style="background-color: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #1b5e20; text-transform: uppercase;">Consumo Real</div>
                    <div style="font-size: 18px; font-weight: 800; color: #1b5e20; margin-top: 4px;">{int(dados_dia['consumo'])} ref</div>
                    <div style="font-size: 10px; color: #2e7d32; margin-top: 2px;">Apurado</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #fff8e1; border: 1px solid #ffe082; border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #f57f17; text-transform: uppercase;">Consumo Real</div>
                    <div style="font-size: 15px; font-weight: 700; color: #e65100; margin-top: 4px;">Pendente</div>
                    <div style="font-size: 10px; color: #f57f17; margin-top: 2px;">Pós-almoço</div>
                </div>
                """, unsafe_allow_html=True)
                
        with c_v2:
            if dados_dia['planejado_nutri'] is not None:
                st.markdown(f"""
                <div style="background-color: #e3f2fd; border: 1px solid #90caf9; border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #0d47a1; text-transform: uppercase;">Estimativa Nutrição</div>
                    <div style="font-size: 18px; font-weight: 800; color: #0d47a1; margin-top: 4px;">{int(dados_dia['planejado_nutri'])} ref</div>
                    <div style="font-size: 10px; color: #1565c0; margin-top: 2px;">Informada</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #fff8e1; border: 1px solid #ffe082; border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #f57f17; text-transform: uppercase;">Estimativa Nutrição</div>
                    <div style="font-size: 15px; font-weight: 700; color: #e65100; margin-top: 4px;">Não informada</div>
                    <div style="font-size: 10px; color: #f57f17; margin-top: 2px;">Pendente</div>
                </div>
                """, unsafe_allow_html=True)
                
        with c_v3:
            prato_txt = dados_dia["prato_principal"] or ""
            if prato_txt:
                st.markdown(f"""
                <div style="background-color: #f1f8e9; border: 1px solid #c5e1a5; border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #33691e; text-transform: uppercase;">Cardápio Oficial</div>
                    <div style="font-size: 13px; font-weight: 700; color: #33691e; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{prato_txt.title()}">{prato_txt.title()}</div>
                    <div style="font-size: 10px; color: #558b2f; margin-top: 2px;">{dados_dia['proteina_tipo'].title()}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #fff8e1; border: 1px solid #ffe082; border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #f57f17; text-transform: uppercase;">Cardápio Oficial</div>
                    <div style="font-size: 14px; font-weight: 700; color: #e65100; margin-top: 4px;">Não cadastrado</div>
                    <div style="font-size: 10px; color: #f57f17; margin-top: 2px;">Sem cardápio</div>
                </div>
                """, unsafe_allow_html=True)
                
        with c_v4:
            st.markdown(f"""
            <div style="background-color: #e0f7fa; border: 1px solid #80deea; border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 11px; font-weight: 700; color: #006064; text-transform: uppercase;">Meteorologia</div>
                <div style="font-size: 16px; font-weight: 800; color: #006064; margin-top: 4px;">{dados_dia['temperature_2m_max']:.1f}°C</div>
                <div style="font-size: 10px; color: #00838f; margin-top: 2px;">Chuva: {dados_dia['precipitation_sum']:.1f} mm</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        col_auth_msg, col_auth_lock = st.columns([0.76, 0.24])
        with col_auth_msg:
            st.success(f"**Acesso de Operador Habilitado:** Conectado como **{st.session_state['usuario_nome']}**.")
        with col_auth_lock:
            if st.button("🔒 Bloquear Edição / Sair", use_container_width=True, key="btn_logout_cadastrar"):
                st.session_state["autenticado"] = False
                st.session_state["usuario_nome"] = "Visitante (Acesso Público)"
                st.session_state["usuario_perfil"] = "visitante"
                st.rerun()
        
        with st.expander("De onde vêm os dados exibidos nesta tela? (Clique para saber mais)", expanded=False):
            st.markdown("""
            **Hierarquia de Dados e Rastreabilidade Operacional:**
            1. **Base Histórica Consolidada (Silver):** Os dados de datas anteriores são lidos diretamente da tabela fato integrada (`silver_fato_diaria.parquet`), unindo calendário letivo, consumo real apurado, estimativas e clima histórico.
            2. **Camada de Produção Dinâmica:** Qualquer edição ou novo dado inserido por você nesta tela é armazenado em `dados/producao/edicoes_operacionais.parquet` e sobreposto em tempo real. Isso garante que a base bruta de pesquisa permaneça intacta enquanto suas rotinas operacionais funcionam dinamicamente.
            3. **Meteorologia via Satélite:** Se a data selecionada ainda não constar no arquivo histórico, o sistema consulta a previsão em tempo real diretamente da API oficial do Open-Meteo.
            """)
        
        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            data_sel = st.date_input("Selecione a Data para Edição:", value=date.today(), key="data_picker_aba4")
            
        data_iso = data_sel.strftime("%Y-%m-%d")
        data_fmt = data_sel.strftime("%d/%m/%Y")
        dados_dia = consultar_dados_completos_data(data_iso)
        
        with col_d2:
            if dados_dia["encontrado_base"]:
                st.info(f"**{data_fmt} ({dados_dia['dia_semana']}) já cadastrada na base.** Você pode revisar e alterar os dados abaixo.")
            else:
                st.success(f"**Nova Data ({data_fmt} / {dados_dia['dia_semana']}).** Não há registros anteriores; preencha os campos para salvar.")
                
        # Painel de Diagnóstico em Tempo Real da Data Selecionada
        st.markdown("##### Diagnóstico do Registro Desta Data")
        c_diag1, c_diag2, c_diag3, c_diag4 = st.columns(4)
        with c_diag1:
            if dados_dia['consumo'] is not None:
                st.markdown(f"""
                <div style="background-color: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #1b5e20; text-transform: uppercase;">Consumo Real</div>
                    <div style="font-size: 20px; font-weight: 800; color: #1b5e20; margin-top: 4px;">{int(dados_dia['consumo'])} ref</div>
                    <div style="font-size: 11px; color: #2e7d32; margin-top: 2px;">Apurado no Almoço</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #fff8e1; border: 1px solid #ffe082; border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #f57f17; text-transform: uppercase;">Consumo Real</div>
                    <div style="font-size: 16px; font-weight: 700; color: #e65100; margin-top: 4px;">Pendente</div>
                    <div style="font-size: 11px; color: #f57f17; margin-top: 2px;">Digitar após as 14h</div>
                </div>
                """, unsafe_allow_html=True)
                
        with c_diag2:
            if dados_dia['planejado_nutri'] is not None:
                st.markdown(f"""
                <div style="background-color: #e3f2fd; border: 1px solid #90caf9; border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #0d47a1; text-transform: uppercase;">Estimativa Nutrição</div>
                    <div style="font-size: 20px; font-weight: 800; color: #0d47a1; margin-top: 4px;">{int(dados_dia['planejado_nutri'])} ref</div>
                    <div style="font-size: 11px; color: #1565c0; margin-top: 2px;">Previsão Inicial Salva</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #fff8e1; border: 1px solid #ffe082; border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #f57f17; text-transform: uppercase;">Estimativa Nutrição</div>
                    <div style="font-size: 16px; font-weight: 700; color: #e65100; margin-top: 4px;">Não Informada</div>
                    <div style="font-size: 11px; color: #f57f17; margin-top: 2px;">Planejamento em branco</div>
                </div>
                """, unsafe_allow_html=True)
                
        with c_diag3:
            prato_txt = dados_dia["prato_principal"] or ""
            if prato_txt:
                st.markdown(f"""
                <div style="background-color: #f1f8e9; border: 1px solid #c5e1a5; border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #33691e; text-transform: uppercase;">Cardápio Oficial</div>
                    <div style="font-size: 14px; font-weight: 700; color: #33691e; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{prato_txt.title()}">{prato_txt.title()}</div>
                    <div style="font-size: 11px; color: #558b2f; margin-top: 2px;">Proteína: {dados_dia['proteina_tipo'].title()}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #ffebee; border: 1px solid #ffcdd2; border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #c62828; text-transform: uppercase;">Cardápio Oficial</div>
                    <div style="font-size: 15px; font-weight: 700; color: #b71c1c; margin-top: 4px;">Não Cadastrado</div>
                    <div style="font-size: 11px; color: #c62828; margin-top: 2px;">Cardápio em branco</div>
                </div>
                """, unsafe_allow_html=True)
                
        with c_diag4:
            st.markdown(f"""
            <div style="background-color: #e0f7fa; border: 1px solid #80deea; border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 11px; font-weight: 700; color: #006064; text-transform: uppercase;">Meteorologia</div>
                <div style="font-size: 18px; font-weight: 800; color: #006064; margin-top: 4px;">{dados_dia['temperature_2m_max']:.1f}°C</div>
                <div style="font-size: 11px; color: #00838f; margin-top: 2px;">Chuva: {dados_dia['precipitation_sum']:.1f} mm ({dados_dia.get('clima_fonte', 'Satélite')})</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Alertas operacionais sobre a data
        if dados_dia["consumo"] is None and not dados_dia["prato_principal"]:
            st.warning(f"**Atenção:** A data **{data_fmt}** ainda não possui nem cardápio nem consumo apurado. Preencha os campos abaixo para alimentar o sistema.")
        elif dados_dia["consumo"] is None:
            st.info(f"**Cardápio cadastrado para {data_fmt}.** O consumo real está pendente e deverá ser registrado após as 14h do dia.")
        else:
            st.success(f"**Registro completo para {data_fmt}.** Os dados existentes foram carregados abaixo nos campos do formulário para edição.")
            
        st.markdown("---")
        st.markdown("##### Formulário de Edição Operacional")
        st.caption("🛡️ **Modo de Simulação Ativo:** Os valores salvos serão aplicados dinamicamente no painel durante a sua navegação para demonstrar o recálculo dos algoritmos. Nenhuma base oficial da escola é sobrescrita permanentemente.")
        
        # Formulário com key dinâmica vinculada à data selecionada (data_iso)
        with st.form(f"form_edicao_{data_iso}"):
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                st.markdown("###### Refeições e Contagem")
                val_cons_init = int(dados_dia['consumo']) if dados_dia['consumo'] is not None else 0
                novo_consumo = st.number_input(
                    "Consumo Real Verificado (Refeições Servidas)",
                    min_value=0, max_value=500,
                    value=val_cons_init,
                    step=1,
                    key=f"ed_consumo_{data_iso}",
                    help="Quantidade real de refeições servidas no dia aos estudantes (deixe 0 se ainda for apurar)."
                )
                
                val_nutri_init = int(dados_dia['planejado_nutri']) if dados_dia['planejado_nutri'] is not None else 180
                novo_planejado_nutri = st.number_input(
                    "Quantitativo Estimado Inicial pela Nutrição",
                    min_value=0, max_value=500,
                    value=val_nutri_init,
                    step=1,
                    key=f"ed_nutri_{data_iso}",
                    help="Estimativa inicial da nutricionista responsável."
                )
                
                is_letivo = st.checkbox(
                    "Dia Letivo Regular com Aulas",
                    value=dados_dia["is_dia_letivo"],
                    key=f"ed_letivo_{data_iso}"
                )
                
            with c_f2:
                st.markdown("###### Cardápio do Dia")
                val_prato_init = dados_dia["prato_principal"] or ""
                prato_nome = st.text_input(
                    "Nome do Prato Principal",
                    value=val_prato_init,
                    placeholder="Ex: Iscas de Frango Grelhado",
                    key=f"ed_prato_{data_iso}"
                )
                
                opcoes_proteina = ["Bovina", "Frango", "Suína", "Peixe", "Ovo / Vegetariana"]
                idx_prot = 1
                for i, p in enumerate(opcoes_proteina):
                    if p.lower() in dados_dia["proteina_tipo"].lower():
                        idx_prot = i
                        break
                proteina_sel = st.selectbox(
                    "Tipo de Proteína Principal",
                    opcoes_proteina,
                    index=idx_prot,
                    key=f"ed_prot_{data_iso}"
                )
                
                st.markdown("###### Meteorologia (Previsão Local)")
                col_w1, col_w2 = st.columns(2)
                with col_w1:
                    chuva_val = st.number_input(
                        "Chuva Prevista (mm)",
                        min_value=0.0, max_value=150.0,
                        value=float(dados_dia["precipitation_sum"]),
                        step=0.5,
                        key=f"ed_chuva_{data_iso}"
                    )
                with col_w2:
                    temp_val = st.number_input(
                        "Temperatura Máx (°C)",
                        min_value=0.0, max_value=45.0,
                        value=float(dados_dia["temperature_2m_max"]),
                        step=0.5,
                        key=f"ed_temp_{data_iso}"
                    )
                    
            st.markdown("---")
            st.markdown("###### Confirmação Obrigatória de Gravação")
            tipo_acao_txt = "atualizar os dados já existentes" if dados_dia["encontrado_base"] else "cadastrar este novo registro"
            confirma_salvar = st.checkbox(
                f"Confirmo que revisei os dados acima e desejo **{tipo_acao_txt}** (em modo de simulação/teste) para o dia **{data_fmt} ({dados_dia['dia_semana']})**.",
                key=f"ed_confirma_{data_iso}"
            )
            
            btn_salvar = st.form_submit_button("Salvar Informações Desta Data", type="primary", use_container_width=True)
            
        if btn_salvar:
            if not confirma_salvar:
                st.warning(f"**Confirmação Obrigatória:** Por favor, marque a caixa de seleção acima confirmando que deseja {tipo_acao_txt} para o dia **{data_fmt}**.")
            else:
                prato_nome_limpo = (prato_nome or "").strip()
                if not prato_nome_limpo:
                    prato_nome_limpo = "Refeição Tradicional do Dia" if is_letivo else "Sem Cardápio"
                    
                dict_salvar = {
                    "consumo": novo_consumo if novo_consumo > 0 else None,
                    "reservas": int(dados_dia['reservas']) if dados_dia['reservas'] is not None else None,
                    "planejado_nutri": novo_planejado_nutri,
                    "is_dia_letivo": is_letivo,
                    "evento_tipo": "Letivo Regular" if is_letivo else "Sem Aula",
                    "prato_principal": prato_nome_limpo,
                    "proteina_tipo": proteina_sel,
                    "preparo_tipo": dados_dia["preparo_tipo"] or "Assado / Grelhado",
                    "precipitation_sum": chuva_val,
                    "temperature_2m_max": temp_val,
                    "observacao": f"Registro operacional salvo por {st.session_state['usuario_nome']} em {datetime.now().strftime('%d/%m/%Y %H:%M')}"
                }
                salvar_edicao_operacional_completa(data_iso, dict_salvar)
                st.success(f"**Sucesso!** As informações do dia **{data_fmt}** foram integradas dinamicamente ao painel para teste e simulação.")
                st.rerun()

        # Auditoria e Transparência: Histórico de Alterações Salvas
        df_ultimas = listar_ultimas_edicoes(10)
        if not df_ultimas.empty:
            st.markdown("---")
            with st.expander("📋 Registros e Alterações Salvas no Sistema (Auditoria)", expanded=False):
                st.caption("Abaixo estão as últimas alterações gravadas no arquivo operacional (`dados/producao/edicoes_operacionais.parquet`):")
                cols_uteis = [c for c in ["timestamp_edicao", "data_iso", "consumo", "planejado_nutri", "prato_principal", "proteina_tipo", "observacao"] if c in df_ultimas.columns]
                st.dataframe(df_ultimas[cols_uteis], use_container_width=True, hide_index=True)
                
                if os.path.exists(CSV_CONSUMO_ATUALIZADO):
                    with open(CSV_CONSUMO_ATUALIZADO, "rb") as f_csv:
                        st.download_button(
                            label="📥 Baixar Dados Completos Atualizados (CSV)",
                            data=f_csv,
                            file_name=f"consumo_atualizado_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            help="Baixar o arquivo consolidado com o histórico completo e as alterações operacionais aplicadas."
                        )

# ==============================================================================
# PÁGINA 6: GUIA PRÁTICO (POP DE ROTINA EM 3 MINUTOS)
# ==============================================================================
elif pagina_ativa == "Guia Prático":
    st.markdown('<div class="story-badge">ETAPA 6 DE 6 • MANUAL OPERACIONAL PADRÃO (POP)</div>', unsafe_allow_html=True)
    st.markdown("### Guia Prático & Procedimento Operacional Padrão (POP)")
    st.markdown('<p class="story-subtitle">Cartilha operacional resumida em 3 minutos para apoio diário da nutricionista, da gestora e da equipe da cozinha terceirizada.</p>', unsafe_allow_html=True)
    
    renderizar_painel_explicativo(
        objetivo_rotina=(
            "**Servir como cartilha rápida de consulta para o dia a dia.** "
            "Ideal para treinar novos funcionários da cozinha, estagiários de nutrição ou consultar dúvidas rápidas."
        ),
        como_foi_feito_ciencia=(
            "**Transposição Tecnológica da Pesquisa:** "
            "Condensa as conclusões dos capítulos 4, 5 e 6 da dissertação em passos operacionais práticos e executáveis."
        ),
        como_usar_passos=(
            "1. **Consulte a rotina de sexta-feira** antes de passar a ordem de preparo.\n"
            "2. **Verifique as ações recomendadas** para dias de chuva forte ou sextas-feiras.\n"
            "3. **Mantenha este guia salvo** como referência operacional do restaurante."
        )
    )

    tab_pop, tab_dissertacao, tab_auditoria = st.tabs([
        "📋 Procedimento Operacional Padrão (POP)",
        "📊 Evidências da Dissertação (2026)",
        "🛡️ Auditoria de Vazamento"
    ])
    
    with tab_pop:
        st.markdown("""
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
        """)
        
    with tab_dissertacao:
        st.markdown("#### 📊 Evidências Empíricas da Dissertação de Mestrado (DSR)")
        st.markdown("**Síntese Geral: Período Retrospectivo (Fev-Ago/2026) vs. Prospectivo (Set-Out/2026, modelo congelado em 31/08/2026)**")
        
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
        st.markdown("##### 🌊 Enchentes de Setembro de 2026")
        col_en1, col_en2 = st.columns([1.2, 0.8])
        with col_en1:
            caminho_tab07 = os.path.join(ROOT_DIR, "resultados", "tabelas_dissertacao", "tabela_07_desempenho_prospectivo_set_out.csv")
            if os.path.exists(caminho_tab07):
                df_tab07 = pd.read_csv(caminho_tab07, sep=";")
                df_enc = df_tab07[df_tab07["Recorte"].str.startswith("Dias de enchente")].set_index("Configuracao")
                e_nutri = df_enc.loc["Nutricionista Institucional"]
                e_ia = df_enc.loc["ExtraTrees Congelado em 31/08"]
                st.markdown(f"""
                Nos 5 dias de alagamento, nenhum método antecipou o evento:
                - **Nutricionista:** MAE de {str(e_nutri['MAE']).replace('.', ',')} refeições, com {int(e_nutri['Sobra_Total_ref'])} refeições de sobra.
                - **ExtraTrees congelado em 31/08:** MAE de {str(e_ia['MAE']).replace('.', ',')} refeições, com {int(e_ia['Sobra_Total_ref'])} refeições de sobra.
                - **Diferença:** {f"{e_nutri['MAE'] - e_ia['MAE']:.2f}".replace('.', ',')} refeições por dia, com previsões ainda muito acima do consumo real.
                """)
        with col_en2:
            p_fig25 = os.path.join(ROOT_DIR, "resultados", "figuras", "dissertacao", "figura_25_confronto_diario_setembro_enchentes_ia_vs_nutri.png")
            if os.path.exists(p_fig25):
                st.image(p_fig25, caption="Figura 25: Dias sem Evento Registrado vs. Dias de Enchente")

    with tab_auditoria:
        st.markdown("#### 🛡️ Auditoria de Vazamento Temporal")
        st.markdown("Testes automatizados da avaliação da pesquisa: treino, seleção de dias e atributos.")
        
        caminho_relatorio = os.path.join(ROOT_DIR, "resultados", "auditoria", "relatorio_teste_vazamento_modelos.json")
        if os.path.exists(caminho_relatorio):
            with open(caminho_relatorio, "r", encoding="utf-8") as f_rel:
                rel_json = json.load(f_rel)
                
            st.success(f"✅ **STATUS GERAL DA BASE:** {rel_json.get('status_geral', 'Não executado')}")
            st.caption(f"Última auditoria executada em: {rel_json.get('data_auditoria', '01/11/2026')}")
            
            for teste in rel_json.get("testes", []):
                with st.container(border=True):
                    st.markdown(f"**{teste.get('teste')}**")
                    if teste.get("sucesso"):
                        st.markdown("🟢 **Resultado:** APROVADO NO TESTE")
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
                    st.success(f"Bateria de testes finalizada: {rel_novo['status_geral']}.")
                    st.rerun()
                except Exception as e_aud:
                    st.error(f"Erro ao executar auditoria: {e_aud}")

# ==============================================================================
# RODAPÉ ACADÊMICO INSTITUCIONAL (DSR / PPG-GTSP)
# ==============================================================================
st.markdown("""
<div style="margin-top: 50px; padding: 24px 16px; border-top: 1px solid #e2e8f0; text-align: center; font-size: 12px; color: #64748b; background-color: #ffffff; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
    <div style="font-weight: 700; color: #0f172a; margin-bottom: 4px;">Restaurante Estudantil — Painel de Apoio à Decisão</div>
    <div>Dissertação de Mestrado Profissional • Metodologia Design Science Research (DSR) • Ciclo 2026</div>
    <div style="margin-top: 4px;">Programa de Pós-Graduação em Gestão e Tecnologia em Sistemas Produtivos (PPG-GTSP)</div>
    <div style="margin-top: 6px; font-size: 11px; color: #94a3b8;">
        Mestrando: <b>Michel Silva de Souza</b> • Orientador: <b>Prof. Dr. Carlos Hideo Arima</b> • Coorientador: <b>Prof. Dr. Rafael Nobre Orsi</b>
    </div>
</div>
""", unsafe_allow_html=True)
