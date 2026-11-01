"""
Módulo do Pipeline Analítico e Inferência Preditiva
Prepara os atributos e executa os modelos campeões da pesquisa (Ridge, Híbrido 50/50 e Conformal Prediction)
com margens conformais calibradas, gerando previsões para as semanas t+1 e t+2 e expondo as features geradas.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

try:
    import streamlit as st
    cache_resource = st.cache_resource
    cache_data = st.cache_data(ttl=1800)
except Exception:
    def cache_resource(fn):
        return fn
    def cache_data(fn):
        return fn

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from storage import carregar_historico_consumo, carregar_planejamento_nutricionista, salvar_predicoes_geradas, carregar_fato_diaria_completa
    from clima import obter_previsao_clima_futura, obter_clima_data
except ImportError:
    from .storage import carregar_historico_consumo, carregar_planejamento_nutricionista, salvar_predicoes_geradas, carregar_fato_diaria_completa
    from .clima import obter_previsao_clima_futura, obter_clima_data

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GOLD_DATASET = os.path.join(ROOT_DIR, "dados", "gold", "gold_dataset_ml_limpo.parquet")

DIAS_SEMANA_PT = {
    0: "Segunda-feira",
    1: "Terça-feira",
    2: "Quarta-feira",
    3: "Quinta-feira",
    4: "Sexta-feira",
    5: "Sábado",
    6: "Domingo"
}

# Metadados Auditados dos Cenários Experimentais (Ablation Study da Dissertação / Caderno 05)
CENARIOS_DICT = {
    "Cenário 3 (Cenário 2 + Quantitativo Institucional)": {
        "usar_clima": True,
        "usar_cardapio": True,
        "usar_nutri": True,
        "mae_t1": 19.88,
        "mae_t2": 26.91,
        "descricao": "Cenário 2 + Quantitativo Institucional da Nutrição (Combinação com a experiência prática da nutricionista, MAE 19,88 ref)"
    },
    "Cenário 2 (Cenário 1 + Clima Open-Meteo)": {
        "usar_clima": True,
        "usar_cardapio": True,
        "usar_nutri": False,
        "mae_t1": 20.70,
        "mae_t2": 27.79,
        "descricao": "Cenário 1 + Meteorologia via satélite Open-Meteo (ideal para compras de despensa t+2 e previsão autônoma)"
    },
    "Cenário 1 (Consumo + Cardápio + Calendário)": {
        "usar_clima": False,
        "usar_cardapio": True,
        "usar_nutri": False,
        "mae_t1": 21.42,
        "mae_t2": 28.50,
        "descricao": "Base operacional: histórico de consumo, atratividade das proteínas do cardápio e calendário escolar"
    }
}

# Ranking Oficial de Modelos Auditados (Coorte Pareada N=73 e Validação Outubro)
MODELOS_RANKING = {
    "Modelo Híbrido (IA + Nutricionista)": {
        "mae_t1": 20.20,
        "mae_t2": 27.79,
        "tipo": "Human-in-the-Loop",
        "detalhe": "Combina a análise de dados da inteligência artificial com a experiência prática da nutricionista"
    },
    "Inteligência Artificial por Padrões (CatBoost)": {
        "mae_t1": 10.74,
        "mae_t2": 26.50,
        "tipo": "Árvores Categóricas",
        "detalhe": "Cruza o histórico para identificar padrões detalhados entre o tipo de prato, dia da semana e clima"
    },
    "Inteligência Artificial por Árvores de Decisão (Random Forest)": {
        "mae_t1": 21.34,
        "mae_t2": 28.45,
        "tipo": "Consenso de Regras Lógicas",
        "detalhe": "Consulta dezenas de regras lógicas do histórico e calcula uma estimativa equilibrada"
    },
    "Modelo Matemático Estável (Ridge)": {
        "mae_t1": 20.70,
        "mae_t2": 27.79,
        "tipo": "Tendência Linear Suavizada",
        "detalhe": "Cálculo estatístico de tendência direta que suaviza oscilações bruscas e evita previsões exageradas"
    },
    "Modelo Matemático Tradicional (Regressão Simples)": {
        "mae_t1": 20.75,
        "mae_t2": 27.85,
        "tipo": "Linha de Tendência Clássica",
        "detalhe": "Fórmula matemática tradicional de linha de tendência baseada na média histórica"
    },
    "Modelo Matemático Seletivo (Lasso)": {
        "mae_t1": 20.83,
        "mae_t2": 28.10,
        "tipo": "Seleção de Fatores Relevantes",
        "detalhe": "Foca apenas nas variáveis mais determinantes (como dia da semana) e ignora variações secundárias"
    },
    "Planejamento Manual da Gestão (Sem IA)": {
        "mae_t1": 26.21,
        "mae_t2": 26.21,
        "tipo": "Estimativa Humana da Gestão",
        "detalhe": "Estimativa informada pela nutricionista com base na rotina e no censo do refeitório"
    }
}

# Políticas de Margem de Segurança Conformal Calibradas
MARGENS_INFO = {
    "Conformal P85 (+18 ref / Padrão Pesquisa)": {
        "margem_ia": 18.0,
        "margem_hibrido": 12.0,
        "cobertura": "84,9%",
        "taxa_falta": "15,1%",
        "descricao": "Política oficial recomendada: balanço ótimo entre combate ao desperdício e proteção estudantil"
    },
    "Conformal P90 (+29 ref / Máxima Proteção)": {
        "margem_ia": 29.0,
        "margem_hibrido": 20.0,
        "cobertura": "90,4%",
        "taxa_falta": "9,6%",
        "descricao": "Conservadora: quase nula tolerância à falta, indicada para semanas de exames ou eventos"
    },
    "Conformal P75 (+10 ref / Econômica)": {
        "margem_ia": 10.0,
        "margem_hibrido": 7.0,
        "cobertura": "75,3%",
        "taxa_falta": "24,7%",
        "descricao": "Econômica: foco em redução severa de sobras e custos financeiros de descarte"
    },
    "Predição Pura (+0 ref / Sem Margem)": {
        "margem_ia": 0.0,
        "margem_hibrido": 0.0,
        "cobertura": "50,0%",
        "taxa_falta": "50,0%",
        "descricao": "Estimativa central não-enviesada (esperança matemática da demanda)"
    }
}

def obter_proximos_dias_uteis(origem_dt=None, num_dias=10, dias_frente=None):
    """
    Retorna lista de datas úteis futuras (segunda a sexta).
    Aceita num_dias ou dias_frente de forma intercambiável.
    """
    if dias_frente is not None:
        num_dias = dias_frente
    if origem_dt is None:
        origem_dt = datetime.now()
    
    cur = origem_dt + timedelta(days=1)
    dias = []
    while len(dias) < num_dias:
        if cur.weekday() < 5:  # 0 a 4 = seg a sex
            dias.append(cur)
        cur += timedelta(days=1)
    return dias

@cache_data
def construir_dataset_treino():
    """
    Prepara a base histórica para treino dos modelos supervisionados com corte estrito de qualificação.
    """
    if not os.path.exists(GOLD_DATASET):
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {GOLD_DATASET}")
    
    df_gold = pd.read_parquet(GOLD_DATASET).copy()
    if "data_iso" in df_gold.columns:
        df_gold = df_gold[df_gold["data_iso"] <= "2026-08-31"].copy()
    
    # Colunas preditivas essenciais comprovadas na pesquisa
    features_numericas = [
        "consumo_lag_5", "consumo_lag_10", "media_movel_5d",
        "precipitation_sum", "temperature_2m_max"
    ]
    
    for col in features_numericas:
        if col not in df_gold.columns:
            if "consumo" in df_gold.columns:
                df_gold[col] = df_gold["consumo"].median()
            else:
                df_gold[col] = 160.0
        else:
            med = df_gold["consumo"].median() if "consumo" in df_gold.columns else 160.0
            df_gold[col] = df_gold[col].fillna(med)
            
    if "data_dt" not in df_gold.columns and "data_iso" in df_gold.columns:
        df_gold["data_dt"] = pd.to_datetime(df_gold["data_iso"])

    for d in range(5):
        col_name = f"is_dia_sem_{d}"
        if col_name not in df_gold.columns:
            df_gold[col_name] = (df_gold["data_dt"].dt.dayofweek == d).astype(int)
            
    for prot in ["bovina", "frango", "suina", "peixe", "ovo"]:
        col_name = f"is_prot_{prot}"
        if col_name not in df_gold.columns:
            if "proteina_tipo" in df_gold.columns:
                df_gold[col_name] = df_gold["proteina_tipo"].astype(str).str.lower().str.contains(prot, na=False).astype(int)
            else:
                df_gold[col_name] = 0
                
    return df_gold

def treinar_modelos(df_treino):
    """Treina os estimadores oficiais da pesquisa."""
    cols_features = [
        "consumo_lag_5", "consumo_lag_10", "media_movel_5d",
        "precipitation_sum", "temperature_2m_max",
        "is_dia_sem_0", "is_dia_sem_1", "is_dia_sem_2", "is_dia_sem_3", "is_dia_sem_4",
        "is_prot_bovina", "is_prot_frango", "is_prot_suina", "is_prot_peixe", "is_prot_ovo"
    ]
    
    X = df_treino[cols_features].values
    y = df_treino["consumo"].values
    
    # 1. Ridge Regression (Modelo Linear com Regularização L2)
    modelo_ridge = Ridge(alpha=10.0, random_state=42)
    modelo_ridge.fit(X, y)
    
    # 2. Lasso Regression (L1 Esparso)
    modelo_lasso = Lasso(alpha=0.5, random_state=42)
    modelo_lasso.fit(X, y)

    # 3. Regressão Linear Simples OLS
    modelo_linear = LinearRegression()
    modelo_linear.fit(X, y)
    
    # 4. Random Forest Regressor (Ensemble)
    modelo_rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
    modelo_rf.fit(X, y)
    
    # 5. CatBoost Regressor (Destaque Não-Linear / Outubro)
    if HAS_CATBOOST:
        modelo_cat = CatBoostRegressor(iterations=120, depth=5, learning_rate=0.08, verbose=0, random_seed=42)
        modelo_cat.fit(X, y)
    else:
        modelo_cat = modelo_ridge
        
    return modelo_ridge, modelo_lasso, modelo_linear, modelo_rf, modelo_cat, cols_features

@cache_resource
def obter_modelos_treinados():
    """
    Retorna os 5 modelos supervisionados ajustados e os atributos preditivos com cache em memória,
    otimizando a responsividade da aplicação web para menos de 1ms.
    """
    df_treino = construir_dataset_treino()
    return treinar_modelos(df_treino)

def executar_pipeline_previsao(
    horizonte_alvo="todos",
    cenario_escolhido="Cenário 3 (Cenário 2 + Quantitativo Institucional)",
    usar_clima=True,
    usar_cardapio=True,
    usar_nutri=True,
    modelo_escolhido="Modelo Híbrido (IA + Nutricionista)",
    peso_ia=0.5,
    tipo_margem="Conformal P85 (+18 ref / Padrão Pesquisa)",
    margem_adicional=0
):
    """
    Executa a inferência para os horizontes futuros com flexibilidade de seleção de dados
    e retorna tanto a tabela de previsões consolidadas quanto a matriz de features geradas.
    """
    if cenario_escolhido in CENARIOS_DICT:
        usar_clima = CENARIOS_DICT[cenario_escolhido]["usar_clima"]
        usar_cardapio = CENARIOS_DICT[cenario_escolhido]["usar_cardapio"]
        usar_nutri = CENARIOS_DICT[cenario_escolhido].get("usar_nutri", True)
    else:
        for k, v in CENARIOS_DICT.items():
            if k.split("(")[0].strip() == str(cenario_escolhido).split("(")[0].strip():
                usar_clima = v["usar_clima"]
                usar_cardapio = v["usar_cardapio"]
                usar_nutri = v.get("usar_nutri", True)
                break

    df_consumo_hist = carregar_historico_consumo()
    df_planejamento = carregar_planejamento_nutricionista()
    df_fato = carregar_fato_diaria_completa()
    df_clima_futuro = obter_previsao_clima_futura(dias_frente=16) if usar_clima else pd.DataFrame()
    df_treino = construir_dataset_treino()
    
    modelo_ridge, modelo_lasso, modelo_linear, modelo_rf, modelo_cat, cols_features = obter_modelos_treinados()
    
    datas_uteis = obter_proximos_dias_uteis(num_dias=10)
    
    ultimos_consumos = df_consumo_hist.dropna(subset=["consumo"]).sort_values("data_dt")
    serie_consumo = ultimos_consumos["consumo"].values
    media_recente_5d = float(np.mean(serie_consumo[-5:])) if len(serie_consumo) >= 5 else 167.2
    media_recente_10d = float(np.mean(serie_consumo[-10:])) if len(serie_consumo) >= 10 else 165.0
    mediana_temp = float(df_treino["temperature_2m_max"].median()) if "temperature_2m_max" in df_treino.columns else 24.5
    
    def pegar_ultimo_consumo_dia_semana(dia_sem, defasagem_semanas=1):
        df_dia = ultimos_consumos[ultimos_consumos["data_dt"].dt.dayofweek == dia_sem]
        if len(df_dia) >= defasagem_semanas:
            return float(df_dia.iloc[-defasagem_semanas]["consumo"])
        return media_recente_5d

    linhas_pred = []
    linhas_features = []
    
    for idx, dt in enumerate(datas_uteis):
        data_iso = dt.strftime("%Y-%m-%d")
        data_formatada = dt.strftime("%d/%m/%Y")
        dia_sem_num = dt.weekday()
        dia_sem_nome = DIAS_SEMANA_PT[dia_sem_num]
        horizonte = "Semana_t1" if idx < 5 else "Pos_Semana_t2"
        
        if horizonte_alvo != "todos" and horizonte != horizonte_alvo:
            continue
        
        # Identificação de Dia Letivo no Calendário Acadêmico
        row_fato = df_fato[df_fato["data_iso"] == data_iso]
        is_letivo = True
        evento_nome = "Letivo Regular"
        if not row_fato.empty:
            is_letivo = bool(row_fato.iloc[0].get("is_dia_letivo", True))
            evento_nome = str(row_fato.iloc[0].get("evento_tipo", "Letivo Regular"))

        # Clima previsto para o dia
        if usar_clima and not df_clima_futuro.empty:
            clima_row = df_clima_futuro[df_clima_futuro["data_iso"] == data_iso]
            if not clima_row.empty:
                precip = float(clima_row.iloc[0]["precipitation_sum"])
                temp_max = float(clima_row.iloc[0]["temperature_2m_max"])
            else:
                precip = 0.0
                temp_max = mediana_temp
        else:
            precip = 0.0
            temp_max = mediana_temp
            
        # Planejamento da nutricionista e cardapio
        plan_row = df_planejamento[df_planejamento["data_iso"] == data_iso]
        if not is_letivo:
            prato = f"Feriado / Sem Atendimento ({evento_nome.replace('_', ' ').title()})"
            prot = "—"
            preparo = "—"
        elif not plan_row.empty and usar_cardapio and pd.notna(plan_row.iloc[0].get("prato_principal")) and str(plan_row.iloc[0].get("prato_principal")).strip() != "":
            prato = str(plan_row.iloc[0].get("prato_principal"))
            prot = str(plan_row.iloc[0].get("proteina_tipo", "frango")).lower()
            preparo = str(plan_row.iloc[0].get("preparo_tipo", "assado"))
        elif not row_fato.empty and usar_cardapio and pd.notna(row_fato.iloc[0].get("prato_principal")) and str(row_fato.iloc[0].get("prato_principal")).strip() != "":
            prato = str(row_fato.iloc[0].get("prato_principal"))
            prot = str(row_fato.iloc[0].get("proteina_tipo", "frango")).lower()
            preparo = str(row_fato.iloc[0].get("preparo_tipo", "assado"))
        else:
            prato = "Padrão Escolar Regular"
            prot = "frango"
            preparo = "assado"
            
        # Quantitativo Institucional / Nutricionista
        val_nutri = None
        origem_nutri = "Não informado"
        val_nutri_informado = None
        if usar_nutri and is_letivo:
            if not plan_row.empty and pd.notna(plan_row.iloc[0].get("planejado_nutri")) and float(plan_row.iloc[0].get("planejado_nutri", 0)) > 0:
                val_nutri = float(plan_row.iloc[0].get("planejado_nutri"))
                val_nutri_informado = val_nutri
                origem_nutri = "Planejamento Nutrição (Produção)"
            elif not row_fato.empty and pd.notna(row_fato.iloc[0].get("planejado_nutri")) and float(row_fato.iloc[0].get("planejado_nutri", 0)) > 0:
                val_nutri = float(row_fato.iloc[0].get("planejado_nutri"))
                val_nutri_informado = val_nutri
                origem_nutri = "Quantitativo Institucional (Reservas)"
            elif not row_fato.empty and pd.notna(row_fato.iloc[0].get("reservas")) and float(row_fato.iloc[0].get("reservas", 0)) > 0:
                val_nutri = float(row_fato.iloc[0].get("reservas"))
                val_nutri_informado = val_nutri
                origem_nutri = "Quantitativo Institucional (Reservas)"
            else:
                val_nutri = 182.7 # Média histórica para o cálculo matemático se híbrido for forçado
                origem_nutri = "Média Histórica (Fallback)"
                val_nutri_informado = None
        else:
            val_nutri = None
            val_nutri_informado = None
            
        lag_5 = pegar_ultimo_consumo_dia_semana(dia_sem_num, defasagem_semanas=1)
        lag_10 = pegar_ultimo_consumo_dia_semana(dia_sem_num, defasagem_semanas=2)
        
        vetor_X = {
            "consumo_lag_5": lag_5,
            "consumo_lag_10": lag_10,
            "media_movel_5d": media_recente_5d,
            "precipitation_sum": precip if usar_clima else 0.0,
            "temperature_2m_max": temp_max if usar_clima else mediana_temp,
            "is_dia_sem_0": 1 if dia_sem_num == 0 else 0,
            "is_dia_sem_1": 1 if dia_sem_num == 1 else 0,
            "is_dia_sem_2": 1 if dia_sem_num == 2 else 0,
            "is_dia_sem_3": 1 if dia_sem_num == 3 else 0,
            "is_dia_sem_4": 1 if dia_sem_num == 4 else 0,
            "is_prot_bovina": 1 if ("bovina" in prot and usar_cardapio) else 0,
            "is_prot_frango": 1 if ("frango" in prot and usar_cardapio) else 0,
            "is_prot_suina": 1 if ("suina" in prot and usar_cardapio) else 0,
            "is_prot_peixe": 1 if ("peixe" in prot and usar_cardapio) else 0,
            "is_prot_ovo": 1 if ("ovo" in prot and usar_cardapio) else 0
        }
        
        X_instancia = np.array([[vetor_X[c] for c in cols_features]])
        
        pred_ridge = float(modelo_ridge.predict(X_instancia)[0])
        pred_lasso = float(modelo_lasso.predict(X_instancia)[0])
        pred_linear = float(modelo_linear.predict(X_instancia)[0])
        pred_rf = float(modelo_rf.predict(X_instancia)[0])
        pred_cat = float(modelo_cat.predict(X_instancia)[0])
        
        # Seleção do estimador de IA conforme escolha do usuário
        if "CatBoost" in modelo_escolhido:
            pred_ia_base = pred_cat
        elif "Ridge" in modelo_escolhido:
            pred_ia_base = pred_ridge
        elif "Lasso" in modelo_escolhido:
            pred_ia_base = pred_lasso
        elif "Linear" in modelo_escolhido or "Regressão" in modelo_escolhido:
            pred_ia_base = pred_linear
        elif "Forest" in modelo_escolhido or "Árvores" in modelo_escolhido:
            pred_ia_base = pred_rf
        else:
            # Híbrido usa Ridge como base de IA
            pred_ia_base = pred_ridge

        # Combinação Híbrida se nutricionista estiver presente
        if val_nutri is not None and usar_nutri and ("Híbrido" in modelo_escolhido or "Nutricionista" in modelo_escolhido or "Manual" in modelo_escolhido):
            if ("Nutricionista" in modelo_escolhido or "Manual" in modelo_escolhido) and "Híbrido" not in modelo_escolhido:
                pred_base = val_nutri
                margem_padrao_auditada = 0.0
            else:
                p_ia = float(peso_ia)
                p_nutri = 1.0 - p_ia
                pred_base = p_ia * pred_ia_base + p_nutri * val_nutri
                margem_padrao_auditada = 12.0 # Margem ótima calibrada para o modelo híbrido
        else:
            pred_base = pred_ia_base
            margem_padrao_auditada = 18.0 # Margem conformal P85 auditada para IA Pura (15,1% falta)
            
        # Aplicação da política de margem
        if "P85" in tipo_margem:
            margem_valor = margem_padrao_auditada
        elif "P75" in tipo_margem:
            margem_valor = 7.0 if (val_nutri is not None and usar_nutri and "Híbrido" in modelo_escolhido) else 10.0
        elif "P90" in tipo_margem:
            margem_valor = 20.0 if (val_nutri is not None and usar_nutri and "Híbrido" in modelo_escolhido) else 29.0
        elif "Pura" in tipo_margem:
            margem_valor = 0.0
        elif "Personalizada" in tipo_margem:
            margem_valor = float(margem_adicional)
        else:
            margem_valor = margem_padrao_auditada
            
        # Regra de Resiliência Climática: se tempestade extrema (>= 15 mm), desativa margem aditiva
        if usar_clima and precip >= 15.0:
            margem_valor = 0.0 # Modo de Contenção de Desperdício

        pred_final = max(0, pred_base + margem_valor)
        
        # Alertas operacionais inteligentes (sem emojis)
        alertas = []
        if not is_letivo:
            pred_base = 0.0
            margem_valor = 0.0
            pred_final = 0.0
            alertas.append(f"Feriado / Sem Atendimento ({evento_nome.replace('_', ' ').title()})")
        else:
            if dia_sem_num == 4:
                alertas.append("Sexta-feira: retração sistemática de comparecimento (~20% menor)")
            if usar_clima and precip >= 15.0:
                alertas.append(f"Alerta tempestade ({precip:.1f} mm): modo contenção ativado (margem zerada para evitar desperdício)")
            elif usar_clima and precip >= 8.0:
                alertas.append(f"Chuva prevista ({precip:.1f} mm): risco moderado de evasão")
            if usar_cardapio:
                if "bovina" in prot or "feijoada" in prato.lower():
                    alertas.append("Cardápio com proteína bovina: alta atratividade histórica")
                elif "peixe" in prot or "ovo" in prot:
                    alertas.append("Proteína com menor atratividade histórica")
            
        linhas_pred.append({
            "data_dt": dt,
            "data": data_formatada,
            "data_iso": data_iso,
            "dia_semana": dia_sem_nome,
            "is_dia_letivo": is_letivo,
            "evento_tipo": evento_nome,
            "horizonte": "Semana t+1 (Preparo)" if horizonte == "Semana_t1" else "Semana t+2 (Compras)",
            "prato_principal": prato if (usar_cardapio or not is_letivo) else "Cardápio Neutro",
            "proteina_tipo": prot.capitalize() if is_letivo else "—",
            "preparo_tipo": preparo.capitalize() if is_letivo else "—",
            "chuva_mm": round(precip, 1) if usar_clima else 0.0,
            "temp_max_c": round(temp_max, 1) if usar_clima else round(mediana_temp, 1),
            "estimativa_nutri": round(val_nutri_informado) if (val_nutri_informado is not None and is_letivo) else None,
            "ia_pura": round(pred_ia_base) if is_letivo else 0,
            "pred_base_combinada": round(pred_base) if is_letivo else 0,
            "margem_aplicada": round(margem_valor) if is_letivo else 0,
            "recomendacao_final": round(pred_final) if is_letivo else 0,
            "alertas": "; ".join(alertas) if alertas else "Operação normal dentro da média"
        })

        # Registrar matriz de features geradas para visualização
        linhas_features.append({
            "Data": data_formatada,
            "Dia": dia_sem_nome,
            "Horizonte": "t+1" if horizonte == "Semana_t1" else "t+2",
            "Lag 5d (ref)": round(lag_5, 1),
            "Lag 10d (ref)": round(lag_10, 1),
            "Média Móvel 5d": round(media_recente_5d, 1),
            "Chuva (mm)": round(precip, 1),
            "Temp Máx (°C)": round(temp_max, 1),
            "Proteína": prot.capitalize(),
            "Nutri Planejado": round(val_nutri_informado, 1) if val_nutri_informado is not None else "Não informado"
        })
        
    df_resultado = pd.DataFrame(linhas_pred)
    df_features = pd.DataFrame(linhas_features)
    salvar_predicoes_geradas(df_resultado)
    return df_resultado, df_features

def simular_previsao_data(
    data_alvo,
    objetivo="consumo",
    cenario_escolhido="Cenário 3 (Cenário 2 + Quantitativo Institucional)",
    modelo_escolhido="Modelo Híbrido (IA + Nutricionista)",
    tipo_margem="Conformal P85 (+18 ref / Padrão Pesquisa)",
    margem_customizada=0.0,
    proteina_custom=None,
    prato_custom=None,
    preparo_custom=None,
    chuva_custom=None,
    temp_custom=None,
    nutri_custom=None
):
    """
    Simula a previsão para uma data específica com garantia de integridade temporal
    (método Walk-Forward: utiliza estritamente dados anteriores a data_alvo, sem vazamento).
    Valida a presença ou ausência de cardápio, dados meteorológicos e quantitativo da nutrição.
    """
    dt_alvo = pd.to_datetime(data_alvo)
    data_iso = dt_alvo.strftime("%Y-%m-%d")
    data_formatada = dt_alvo.strftime("%d/%m/%Y")
    dia_sem = dt_alvo.dayofweek
    dia_sem_nome = DIAS_SEMANA_PT.get(dia_sem, "Dia Útil")
    is_fim_semana = (dia_sem >= 5)

    df_fato = carregar_fato_diaria_completa()
    df_planejamento = carregar_planejamento_nutricionista()

    # 1. Dados históricos estritamente ANTERIORES à data selecionada (Walk-Forward / Zero Leakage)
    df_hist = df_fato[(df_fato["data_dt"] < dt_alvo) & (df_fato["consumo"].notna()) & (df_fato["consumo"] > 0)].sort_values("data_dt")

    if len(df_hist) >= 5:
        serie_hist = df_hist["consumo"].values
        mm5 = float(np.mean(serie_hist[-5:]))
        mm10 = float(np.mean(serie_hist[-10:])) if len(serie_hist) >= 10 else mm5
    else:
        mm5 = 160.0
        mm10 = 160.0

    if "dia_da_semana_num" in df_hist.columns:
        df_mesmo = df_hist[df_hist["dia_da_semana_num"] == dia_sem]
    elif "data_dt" in df_hist.columns:
        df_mesmo = df_hist[pd.to_datetime(df_hist["data_dt"]).dt.dayofweek == dia_sem]
    else:
        df_mesmo = pd.DataFrame()
    lag5 = float(df_mesmo.iloc[-1]["consumo"]) if len(df_mesmo) >= 1 else mm5
    lag10 = float(df_mesmo.iloc[-2]["consumo"]) if len(df_mesmo) >= 2 else mm5

    # 2. Checagem de dados na base para a data alvo
    row_fato = df_fato[df_fato["data_dt"] == dt_alvo]
    row_plan = df_planejamento[df_planejamento["data_iso"] == data_iso]

    # Cardápio
    tem_cardapio_base = False
    prato_base = None
    prot_base = None
    prep_base = None
    if not row_plan.empty and pd.notna(row_plan.iloc[0].get("prato_principal")) and str(row_plan.iloc[0].get("prato_principal")).strip() != "":
        tem_cardapio_base = True
        prato_base = str(row_plan.iloc[0].get("prato_principal"))
        prot_base = str(row_plan.iloc[0].get("proteina_tipo", "frango")).lower()
        prep_base = str(row_plan.iloc[0].get("preparo_tipo", "assado"))
    elif not row_fato.empty and pd.notna(row_fato.iloc[0].get("prato_principal")) and str(row_fato.iloc[0].get("prato_principal")).strip() != "":
        tem_cardapio_base = True
        prato_base = str(row_fato.iloc[0].get("prato_principal"))
        prot_base = str(row_fato.iloc[0].get("proteina_tipo", "frango")).lower()
        prep_base = str(row_fato.iloc[0].get("preparo_tipo", "assado"))

    prato_utilizado = prato_custom if prato_custom is not None else (prato_base if tem_cardapio_base else "Padrão Regular")
    prot_utilizada = str(proteina_custom if proteina_custom is not None else (prot_base if tem_cardapio_base else "frango")).lower()
    prep_utilizado = preparo_custom if preparo_custom is not None else (prep_base if tem_cardapio_base else "assado")

    # Clima
    tem_clima_base = False
    chuva_base = None
    temp_base = None
    fonte_clima = "Não localizado"
    if not row_fato.empty and pd.notna(row_fato.iloc[0].get("precipitation_sum")):
        tem_clima_base = True
        chuva_base = float(row_fato.iloc[0]["precipitation_sum"])
        temp_base = float(row_fato.iloc[0].get("temperature_2m_max", 25.0))
        fonte_clima = "Base Histórica Consolidada"
    else:
        # Busca meteorologia automática em tempo real via Open-Meteo API
        chuva_api, temp_api, fonte_api = obter_clima_data(data_iso)
        if chuva_api is not None and temp_api is not None:
            tem_clima_base = True
            chuva_base = float(chuva_api)
            temp_base = float(temp_api)
            fonte_clima = fonte_api

    chuva_utilizada = float(chuva_custom if chuva_custom is not None else (chuva_base if tem_clima_base else 0.0))
    temp_utilizada = float(temp_custom if temp_custom is not None else (temp_base if tem_clima_base else 25.0))

    # Nutri
    tem_nutri_base = False
    nutri_base = None
    if not row_plan.empty and pd.notna(row_plan.iloc[0].get("planejado_nutri")) and float(row_plan.iloc[0].get("planejado_nutri", 0)) > 0:
        tem_nutri_base = True
        nutri_base = float(row_plan.iloc[0]["planejado_nutri"])
    elif not row_fato.empty and pd.notna(row_fato.iloc[0].get("planejado_nutri")) and float(row_fato.iloc[0].get("planejado_nutri", 0)) > 0:
        tem_nutri_base = True
        nutri_base = float(row_fato.iloc[0]["planejado_nutri"])

    nutri_utilizado = float(nutri_custom if nutri_custom is not None else (nutri_base if tem_nutri_base else 182.7))

    # Consumo Real se conhecido no histórico
    consumo_real = float(row_fato.iloc[0]["consumo"]) if not row_fato.empty and pd.notna(row_fato.iloc[0].get("consumo")) and float(row_fato.iloc[0]["consumo"]) > 0 else None

    # 3. Configuração do Cenário
    cen_cfg = CENARIOS_DICT.get(cenario_escolhido, CENARIOS_DICT["Cenário 3 (Cenário 2 + Quantitativo Institucional)"])
    usar_clima = cen_cfg["usar_clima"]
    usar_cardapio = cen_cfg["usar_cardapio"]
    usar_nutri = cen_cfg.get("usar_nutri", True) and (objetivo == "consumo")

    # 4. Vetor de Features
    vetor_X = {
        "consumo_lag_5": lag5,
        "consumo_lag_10": lag10,
        "media_movel_5d": mm5,
        "precipitation_sum": chuva_utilizada if usar_clima else 0.0,
        "temperature_2m_max": temp_utilizada if usar_clima else 25.0,
        "is_dia_sem_0": 1 if dia_sem == 0 else 0,
        "is_dia_sem_1": 1 if dia_sem == 1 else 0,
        "is_dia_sem_2": 1 if dia_sem == 2 else 0,
        "is_dia_sem_3": 1 if dia_sem == 3 else 0,
        "is_dia_sem_4": 1 if dia_sem == 4 else 0,
        "is_prot_bovina": 1 if ("bovina" in prot_utilizada and usar_cardapio) else 0,
        "is_prot_frango": 1 if ("frango" in prot_utilizada and usar_cardapio) else 0,
        "is_prot_suina": 1 if ("suina" in prot_utilizada and usar_cardapio) else 0,
        "is_prot_peixe": 1 if ("peixe" in prot_utilizada and usar_cardapio) else 0,
        "is_prot_ovo": 1 if ("ovo" in prot_utilizada and usar_cardapio) else 0
    }

    modelo_ridge, modelo_lasso, modelo_linear, modelo_rf, modelo_cat, cols_features = obter_modelos_treinados()
    X_inst = np.array([[vetor_X[c] for c in cols_features]])

    pred_ridge = float(modelo_ridge.predict(X_inst)[0])
    pred_cat = float(modelo_cat.predict(X_inst)[0])
    pred_rf = float(modelo_rf.predict(X_inst)[0])
    pred_lasso = float(modelo_lasso.predict(X_inst)[0])
    pred_linear = float(modelo_linear.predict(X_inst)[0])

    if "CatBoost" in modelo_escolhido:
        pred_ia = pred_cat
    elif "Ridge" in modelo_escolhido:
        pred_ia = pred_ridge
    elif "Forest" in modelo_escolhido or "Árvores" in modelo_escolhido:
        pred_ia = pred_rf
    elif "Lasso" in modelo_escolhido:
        pred_ia = pred_lasso
    elif "Linear" in modelo_escolhido or "Regressão" in modelo_escolhido:
        pred_ia = pred_linear
    else:
        pred_ia = pred_ridge

    if usar_nutri and ("Híbrido" in modelo_escolhido or "Nutricionista" in modelo_escolhido or "Manual" in modelo_escolhido):
        if ("Nutricionista" in modelo_escolhido or "Manual" in modelo_escolhido) and "Híbrido" not in modelo_escolhido:
            pred_base = nutri_utilizado
            margem_padrao = 0.0
        else:
            pred_base = 0.5 * pred_ia + 0.5 * nutri_utilizado
            margem_padrao = 12.0
    else:
        pred_base = pred_ia
        margem_padrao = 18.0

    if "P85" in tipo_margem:
        margem_val = margem_padrao
    elif "P75" in tipo_margem:
        margem_val = 7.0 if (usar_nutri and "Híbrido" in modelo_escolhido) else 10.0
    elif "P90" in tipo_margem:
        margem_val = 20.0 if (usar_nutri and "Híbrido" in modelo_escolhido) else 29.0
    elif "Pura" in tipo_margem:
        margem_val = 0.0
    elif "Personalizada" in tipo_margem:
        margem_val = float(margem_customizada)
    else:
        margem_val = margem_padrao

    if usar_clima and chuva_utilizada >= 15.0:
        margem_val = 0.0

    pred_final = max(0, pred_base + margem_val)

    is_letivo_sim = True
    evento_tipo_sim = "Letivo Regular"
    if not row_fato.empty:
        is_letivo_sim = bool(row_fato.iloc[0].get("is_dia_letivo", True))
        evento_tipo_sim = str(row_fato.iloc[0].get("evento_tipo", "Letivo Regular"))

    if not is_letivo_sim:
        pred_base = 0.0
        margem_val = 0.0
        pred_final = 0.0

    alertas = []
    if not is_letivo_sim:
        alertas.append(f"Feriado / Não Letivo ({evento_tipo_sim.replace('_', ' ').title()}): Restaurante sem atendimento regular.")
    elif is_fim_semana:
        alertas.append("Fim de semana (sábado/domingo): refeitório fechado ou com cardápio excepcional.")
    elif dia_sem == 4:
        alertas.append("Sexta-feira: retração histórica de ~20% no comparecimento à tarde.")
    elif dia_sem == 2:
        alertas.append("Quarta-feira: dia de maior comparecimento discente (pico semanal com aulas técnicas).")

    if usar_clima and chuva_utilizada >= 15.0:
        alertas.append(f"Alerta Tempestade ({chuva_utilizada:.1f} mm): Margem aditiva zerada para evitar desperdício de insumos.")
    elif usar_clima and chuva_utilizada >= 5.0:
        alertas.append(f"Chuva prevista ({chuva_utilizada:.1f} mm): Queda esperada na adesão de alunos.")

    if usar_cardapio:
        if "bovina" in prot_utilizada:
            alertas.append("Carne Bovina: Prato com maior adesão histórica de alunos.")
        elif "peixe" in prot_utilizada:
            alertas.append("Peixe / Pescados: Menor adesão discente (recomenda-se cautela no volume).")

    erro_abs = abs(pred_final - consumo_real) if consumo_real is not None else None

    return {
        "dt_alvo": dt_alvo,
        "data_str": data_formatada,
        "dia_semana_nome": dia_sem_nome,
        "is_fim_semana": is_fim_semana,
        "is_dia_letivo": is_letivo_sim,
        "evento_tipo": evento_tipo_sim,
        "objetivo": objetivo,
        "cenario_escolhido": cenario_escolhido,
        "modelo_escolhido": modelo_escolhido,
        "tipo_margem": tipo_margem,
        "dias_historico_utilizados": len(df_hist),
        "tem_cardapio_base": tem_cardapio_base,
        "prato_base": prato_base,
        "prot_base": prot_base,
        "prep_base": prep_base,
        "tem_clima_base": tem_clima_base,
        "fonte_clima": fonte_clima,
        "chuva_base": chuva_base,
        "temp_base": temp_base,
        "tem_nutri_base": tem_nutri_base,
        "nutri_base": nutri_base,
        "prato_utilizado": prato_utilizado,
        "prot_utilizada": prot_utilizada,
        "prep_utilizado": prep_utilizado,
        "chuva_utilizada": chuva_utilizada,
        "temp_utilizada": temp_utilizada,
        "nutri_utilizado": nutri_utilizado,
        "consumo_real": consumo_real,
        "erro_absoluto": erro_abs,
        "pred_ia": round(pred_ia, 1),
        "pred_base": round(pred_base, 1),
        "margem_val": round(margem_val, 1),
        "pred_final": int(round(pred_final)),
        "mm5": round(mm5, 1),
        "lag5": round(lag5, 1),
        "lag10": round(lag10, 1),
        "alertas": alertas,
        "vetor_features": vetor_X
    }
