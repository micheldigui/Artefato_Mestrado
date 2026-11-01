"""Módulo de detecção e classificação determinística de dias atípicos e outliers operacionais."""
from typing import Tuple
import pandas as pd
from .config import SILVER

# Limiares de Detecção em Relação ao Nível Basal
RAZAO_BAIXA = 0.60    # Consumo inferior a 60% do nível basal recente
RAZAO_ALTA = 1.40     # Consumo superior a 140% do nível basal recente

# Janelas Temporais de Referência
JANELA_GERAL = 10     # Últimos 10 dias úteis normais
JANELA_MESMO_DIA = 4  # Últimas 4 ocorrências do mesmo dia da semana
JANELA_RECENTE = 20   # Janela recente para buscar dias normais
MIN_HISTORICO = 8     # Mínimo de histórico para começar a estimar referência
MIN_NORMAIS = 5       # Mínimo de dias normais na janela recente

# Parâmetros de Calendário e Sazonalidade
DIAS_FIM_BLOCO = 4    # Dias finais antes do recesso/férias
DIAS_INICIO_BLOCO = 4 # Primeiros dias após a volta das aulas
TAMANHO_SEQUENCIA = 3 # Dias consecutivos de anomalia
CHUVA_FORTE_MM = 15.0 # Limiar de chuva forte (impacto em deslocamento discente)

# Eventos Rotineiros (onde o consumo esperado permanece dentro do normal do bloco)
EVENTOS_ROTINEIROS = {
    "aula_regular",
    "encontro_institucional",
    "reuniao_apoio_estudantil",
    "prazo_administrativo",
    "formacao",
    "evento_portas_abertas"
}

# Categorias consideradas imprevisíveis no momento da decisão operacional
TIPOS_IMPREVISIVEIS = {
    "evento_externo_registrado",
    "isolado_sem_explicacao",
    "sequencia_de_dias"
}

# Categorias que ensinam os algoritmos preditivos
TRATAMENTOS_NO_TREINO = {
    "normal",
    "alto_mantido",
    "previsivel_calendario"
}


def calcular_referencia_temporal(servidos: pd.DataFrame) -> pd.Series:
    """Calcula a demanda de referência (nível basal) usando estritamente o passado.
    
    Dias já marcados como baixos são expurgados da janela para não rebaixar a
    referência indevidamente. Caso a janela recente possua menos de MIN_NORMAIS,
    todos os dias recentes são considerados para acompanhar tendências reais de queda.
    """
    referencias = []
    eh_baixo = pd.Series(False, index=servidos.index)
    
    for i, linha in enumerate(servidos.itertuples()):
        anteriores = servidos.iloc[:i]
        if len(anteriores) < MIN_HISTORICO:
            referencias.append(float("nan"))
            continue
            
        recentes = anteriores.tail(JANELA_RECENTE)
        normais = recentes[~eh_baixo.loc[recentes.index]]
        if len(normais) < MIN_NORMAIS:
            normais = recentes
            
        mediana_geral = normais["consumo"].tail(JANELA_GERAL).median()
        mediana_mesmo_dia = normais.loc[normais["dia_semana"] == linha.dia_semana, "consumo"].tail(JANELA_MESMO_DIA).median()
        
        valor_referencia = (mediana_geral + mediana_mesmo_dia) / 2.0 if pd.notna(mediana_mesmo_dia) else mediana_geral
        referencias.append(valor_referencia)
        eh_baixo.iloc[i] = (linha.consumo / valor_referencia) < RAZAO_BAIXA
        
    return pd.Series(referencias, index=servidos.index)


def calcular_sequencia_candidatos(servidos: pd.DataFrame) -> pd.Series:
    """Calcula o tamanho do trecho consecutivo de dias anômalos na mesma direção."""
    trecho = servidos["direcao"].ne(servidos["direcao"].shift()).cumsum()
    return servidos.groupby(trecho)["direcao"].transform("size").where(servidos["direcao"].ne(""), 0)


def sugerir_tipo_atipico(linha: pd.Series) -> str:
    """Classifica a causa provável de uma anomalia com base em evidências do calendário e clima."""
    if linha["status_consumo"] == "atipico":
        return "evento_externo_registrado"
    if linha["status_legal"] != "letivo":
        return "nao_letivo_com_consumo"
    if linha["periodo_greve"]:
        return "periodo_de_greve"
    if linha["direcao"] == "baixo" and pd.notna(linha["dias_ate_fim_bloco"]) and linha["dias_ate_fim_bloco"] <= DIAS_FIM_BLOCO:
        return "fim_de_bloco_letivo"
    if pd.notna(linha["dias_desde_inicio_bloco"]) and linha["dias_desde_inicio_bloco"] <= DIAS_INICIO_BLOCO:
        return "inicio_de_bloco_letivo"
    if linha["evento_tipo"] not in EVENTOS_ROTINEIROS:
        return "evento_de_calendario"
    if linha["tamanho_sequencia"] >= TAMANHO_SEQUENCIA:
        return "sequencia_de_dias"
    return "isolado_sem_explicacao"


def detectar_dias_atipicos(fato: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detecta dias fora do padrão comparando consumo com a referência basal."""
    servidos = fato[fato["no_escopo"] & fato["tem_consumo"]].copy()
    servidos["referencia"] = calcular_referencia_temporal(servidos).round(1)
    servidos["razao_referencia"] = (servidos["consumo"] / servidos["referencia"]).round(3)
    
    servidos["direcao"] = ""
    servidos.loc[servidos["razao_referencia"] < RAZAO_BAIXA, "direcao"] = "baixo"
    servidos.loc[servidos["razao_referencia"] > RAZAO_ALTA, "direcao"] = "alto"
    
    servidos["candidato"] = servidos["direcao"].ne("") | servidos["status_consumo"].eq("atipico")
    servidos.loc[servidos["status_consumo"].eq("atipico") & servidos["direcao"].eq(""), "direcao"] = "baixo"
    servidos["tamanho_sequencia"] = calcular_sequencia_candidatos(servidos)
    
    # Impacto de chuva no dia ou acumulada no dia anterior
    fato_idx = fato.set_index("data")
    servidos["chuva_mm_dia_anterior"] = fato_idx["chuva_mm"].shift(1).reindex(servidos["data"]).values
    servidos["chuva_forte"] = (servidos["chuva_mm"] >= CHUVA_FORTE_MM) | (servidos["chuva_mm_dia_anterior"] >= CHUVA_FORTE_MM)
    
    candidatos = servidos[servidos["candidato"]].copy()
    candidatos["tipo_sugerido"] = candidatos.apply(sugerir_tipo_atipico, axis=1)
    candidatos["plano"] = candidatos["reserva_discente"].fillna(candidatos["plano_nutricionista"])
    
    return servidos, candidatos


def definir_tratamento_modelo(linha: pd.Series) -> str:
    """Define se a data deve ou não fazer parte do treinamento dos algoritmos de ML."""
    if linha["status_legal"] != "letivo":
        return "fora_do_calendario"
    if linha["periodo_greve"]:
        return "periodo_de_greve"
    if linha["direcao"] == "alto":
        return "alto_mantido"
    if linha["direcao"] != "baixo":
        return "normal"
    return "isolado_sem_explicacao" if linha["tipo_sugerido"] in TIPOS_IMPREVISIVEIS else "previsivel_calendario"


def classificar_dias(servidos: pd.DataFrame, candidatos: pd.DataFrame) -> pd.DataFrame:
    """Gera a tabela final de classificação e elegibilidade para treino de cada data."""
    dias = servidos.merge(candidatos[["data", "tipo_sugerido"]], on="data", how="left")
    dias["tratamento"] = dias.apply(definir_tratamento_modelo, axis=1)
    dias["usar_no_treino"] = dias["tratamento"].isin(TRATAMENTOS_NO_TREINO)
    return dias[["data", "consumo", "referencia", "razao_referencia", "direcao", "tipo_sugerido", "tratamento", "usar_no_treino"]]
