"""Mapeamento de terminologia: conversão entre termos técnicos e operacionais."""

NOMES_HORIZONTES = {
    "t1": "Preparo (Semana Seguinte)",
    "t2": "Compras (Duas Semanas à Frente)"
}

NOMES_CENARIOS = {
    "cenario_1": "Cenário 1: Histórico e Calendário",
    "cenario_2": "Cenário 2: Inclusão do Cardápio",
    "cenario_3": "Cenário 3: Inclusão do Clima",
    "cenario_4": "Cenário 4: Integração com Nutricionista / Reservas",
}

NOMES_MODELOS = {
    "baseline_recente": "Média Móvel Recente (Baseline 10d)",
    "baseline_mesmo_dia": "Histórico do Mesmo Dia da Semana",
    "regressao_linear": "Regressão Linear Múltipla",
    "ridge": "Regressão Regularizada Ridge",
    "lasso": "Regressão Regularizada Lasso",
    "extra_trees": "ExtraTrees Regressor (Árvores Extremamente Aleatórias)",
    "random_forest": "Random Forest Regressor",
    "gradient_boosting": "Gradient Boosting Regressor",
    "catboost": "CatBoost Regressor",
    "xgboost": "XGBoost Regressor",
    "lightgbm": "LightGBM Regressor",
    "hibrido_adaptativo": "Modelo Híbrido Adaptativo (IA + Nutricionista)"
}

NOMES_TRATAMENTOS = {
    "normal": "Dia Típico (Treinamento)",
    "previsivel_calendario": "Queda Prevista por Calendário (Treinamento)",
    "alto_mantido": "Pico de Consumo Mantido (Treinamento)",
    "isolado_sem_explicacao": "Outlier Imprevisível (Isolado do Treinamento)",
    "periodo_de_greve": "Período de Greve (Isolado do Treinamento)",
    "evento_externo_registrado": "Força Maior / Evento Externo (Isolado)",
    "fora_do_calendario": "Refeição Fora do Calendário Letivo"
}

NOMES_DIAS_SEMANA = {
    0: "Segunda-feira",
    1: "Terça-feira",
    2: "Quarta-feira",
    3: "Quinta-feira",
    4: "Sexta-feira",
    5: "Sábado",
    6: "Domingo"
}

def obter_nome_modelo(chave: str) -> str:
    return NOMES_MODELOS.get(chave, chave.replace("_", " ").title())

def obter_nome_cenario(chave: str) -> str:
    return NOMES_CENARIOS.get(chave, chave)

def obter_nome_tratamento(chave: str) -> str:
    return NOMES_TRATAMENTOS.get(chave, chave)

