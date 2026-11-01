"""Camada Gold: Engenharia de atributos orientada pelo marco temporal da sexta-feira (Zero Data Leakage)."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from .config import GOLD, HORIZONTES, SILVER
from .atipicos import (
    EVENTOS_ROTINEIROS,
    RAZAO_BAIXA,
    detectar_dias_atipicos,
    classificar_dias
)

# Hiperparâmetros de Janelas e Suavização
JANELA_NIVEL = 10
JANELA_LONGA = 20
JANELA_MESMO_DIA = 4
OCORRENCIAS_MESMO_DIA = 4
ALFA_EXPONENCIAL = 0.2
JANELA_RECENTE = 20
MIN_NORMAIS = 5
MIN_HISTORICO = 10
PESO_PRIOR_CARDAPIO = 5
LIMITES_RAZAO = (0.3, 1.5)
JANELA_EPOCA_DIAS = 7
MIN_DIAS_EPOCA = 4
LIMITE_DISTANCIA_BLOCO = 15
PAUSA_FIM_DE_SEMESTRE = 10

ULTIMA_EMISSAO = "2026-10-30"

ATRIBUTOS_DA_EMISSAO = (
    "nivel_10d", "nivel_20d", "nivel_exponencial", "ultimo_consumo",
    "media_5d", "mediana_5d", "media_10d", "media_20d",
    "desvio_5d", "desvio_10d", "minimo_5d", "maximo_5d",
    "dispersao_recente", "ultimo_consumo_usado"
)

RELATIVOS_AO_NIVEL = (
    "ultimo_consumo", "media_5d", "nivel_exponencial",
    "nivel_bloco_atual", "mesmo_dia_ha_1_ocorrencias", "mesmo_dia_ha_2_ocorrencias"
)


def origem_da_janela(datas: pd.Series, deslocamento: int) -> pd.Series:
    """Calcula a sexta-feira de corte da previsão para a semana-alvo."""
    return datas - pd.to_timedelta(datas.dt.dayofweek + deslocamento, unit="D")


def faixa_do_bloco(t: pd.DataFrame) -> np.ndarray:
    """Classifica a posição relativa do dia dentro do bloco letivo contínuo."""
    return np.select(
        [t["dias_ate_fim_bloco"] <= 3, t["dias_ate_fim_bloco"] <= 7, t["dias_desde_inicio_bloco"] <= 4],
        ["ultimos_dias", "antes_do_fim", "primeiros_dias"],
        default="meio"
    )


def mesma_epoca_ano_passado(datas: pd.Series, serie_razao: pd.Series) -> list:
    """Calcula a razão mediana de consumo no mesmo período do ano letivo anterior."""
    valores = []
    for data in datas:
        centro = data - pd.Timedelta(days=364)
        recorte = serie_razao[centro - pd.Timedelta(days=JANELA_EPOCA_DIAS):centro + pd.Timedelta(days=JANELA_EPOCA_DIAS)]
        valores.append(recorte.median() if len(recorte) >= MIN_DIAS_EPOCA else float("nan"))
    return valores


def efeito_suavizado(tabela: pd.DataFrame, coluna: str) -> dict:
    """Calcula o efeito categórico suavizado com prior empírico Bayesiano."""
    grupo = tabela.groupby(coluna)["razao_referencia"].agg(["sum", "count"])
    return ((grupo["sum"] + PESO_PRIOR_CARDAPIO) / (grupo["count"] + PESO_PRIOR_CARDAPIO)).to_dict()


def resumir_historico_causal(historico: pd.DataFrame) -> dict:
    """Resume a dinâmica recente do refeitório utilizando estritamente dados anteriores à sexta de emissão."""
    if len(historico) < MIN_HISTORICO:
        return None
        
    recentes = historico.tail(JANELA_RECENTE)
    normais = recentes[recentes["razao_referencia"].fillna(1) >= RAZAO_BAIXA]
    if len(normais) < MIN_NORMAIS:
        normais = recentes
        
    valores = normais["consumo"]
    resumo = {
        "nivel_10d": valores.tail(JANELA_NIVEL).median(),
        "nivel_20d": valores.tail(JANELA_LONGA).median(),
        "nivel_exponencial": valores.ewm(alpha=ALFA_EXPONENCIAL).mean().iloc[-1],
        "ultimo_consumo": valores.iloc[-1],
        "media_5d": valores.tail(5).mean(),
        "mediana_5d": valores.tail(5).median(),
        "media_10d": valores.tail(10).mean(),
        "media_20d": valores.tail(20).mean(),
        "desvio_5d": valores.tail(5).std(),
        "desvio_10d": valores.tail(10).std(),
        "minimo_5d": valores.tail(5).min(),
        "maximo_5d": valores.tail(5).max(),
        "dispersao_recente": (valores.tail(JANELA_NIVEL) - valores.tail(JANELA_NIVEL).median()).abs().median(),
        "ultimo_consumo_usado": historico["data"].iloc[-1],
    }
    
    # Efeito do mesmo dia da semana
    semanais = historico[historico["razao_referencia"].fillna(1) >= RAZAO_BAIXA].groupby("dia_semana")["consumo"]
    resumo["mesmo_dia"] = semanais.apply(lambda s: s.tail(JANELA_MESMO_DIA).median()).to_dict()
    resumo["ocorrencias"] = semanais.apply(lambda s: s.tail(OCORRENCIAS_MESMO_DIA).tolist()[::-1]).to_dict()
    
    # Efeito de bloco letivo
    normais_bloco = historico[historico["razao_referencia"].fillna(1) >= RAZAO_BAIXA].groupby("bloco")["consumo"]
    resumo["nivel_bloco"] = normais_bloco.median().to_dict()
    resumo["dias_bloco"] = normais_bloco.size().to_dict()
    
    # Efeito do Cardápio via Suavização Bayesiana
    base_cardapio = historico[historico["razao_referencia"].notna() & (historico["razao_referencia"] >= RAZAO_BAIXA)]
    for col in ("proteina", "preparo", "prato_padronizado"):
        resumo[col] = efeito_suavizado(base_cardapio, col)
        
    # Efeito de eventos acadêmicos
    com_razao = historico[historico["razao_referencia"].notna()]
    resumo["evento_tipo"] = efeito_suavizado(com_razao, "evento_tipo")
    
    limitada = com_razao.assign(
        razao_referencia=com_razao["razao_referencia"].clip(*LIMITES_RAZAO),
        faixa_bloco=faixa_do_bloco(com_razao)
    )
    resumo["faixa_bloco"] = efeito_suavizado(limitada, "faixa_bloco")
    resumo["serie_razao"] = com_razao.set_index("data")["razao_referencia"]
    
    return resumo


def pausa_depois_de_cada_bloco(fato: pd.DataFrame, util: pd.DataFrame) -> dict:
    """Calcula a duração da pausa (dias úteis sem aula) que sucede cada bloco letivo."""
    pausas = {}
    for bloco, dias in fato[fato["bloco"].notna()].groupby("bloco"):
        letivo = util[util["data"] > dias["data"].max()]["status_legal"].eq("letivo").to_numpy()
        pausas[bloco] = float(letivo.argmax()) if letivo.any() else np.nan
    return pausas


def preparar_alvos(fato: pd.DataFrame) -> pd.DataFrame:
    """Identifica todos os dias de aula previstos no calendário letivo."""
    alvos = fato[fato["no_escopo"] & fato["status_legal"].eq("letivo")].copy()
    util = fato[fato["no_escopo"]].copy()
    letivo = util["status_legal"].eq("letivo")
    
    util["vespera_de_pausa"] = letivo & ~letivo.shift(-1, fill_value=True)
    util["pos_pausa"] = letivo & ~letivo.shift(1, fill_value=True)
    util["dias_letivos_na_semana"] = letivo.groupby(util["data"].dt.to_period("W")).transform("sum")
    sem_aula = (~letivo).astype(int)
    util["dias_sem_aula_antes"] = sem_aula.groupby(letivo.cumsum()).cumsum().shift(1).fillna(0)
    
    alvos["pausa_apos_bloco"] = alvos["bloco"].map(pausa_depois_de_cada_bloco(fato, util))
    colunas_util = ["data", "vespera_de_pausa", "pos_pausa", "dias_letivos_na_semana", "dias_sem_aula_antes"]
    return alvos.merge(util[colunas_util], on="data")


def montar_janela_temporal(alvos: pd.DataFrame, servidos: pd.DataFrame, deslocamento: int) -> pd.DataFrame:
    """Constrói atributos vinculados a cada sexta-feira de emissão sem usar o futuro."""
    alvos = alvos.copy()
    alvos["origem"] = origem_da_janela(alvos["data"], deslocamento)
    alvos = alvos[alvos["origem"] <= ULTIMA_EMISSAO]
    linhas = []
    
    for origem, grupo in alvos.groupby("origem"):
        # Corte estrito: somente dados com data < origem
        resumo = resumir_historico_causal(servidos[servidos["data"] < origem])
        if resumo is None:
            continue
            
        g = grupo.copy()
        for chave in ATRIBUTOS_DA_EMISSAO:
            g[chave] = resumo[chave]
            
        g["nivel_mesmo_dia"] = g["dia_semana"].map(resumo["mesmo_dia"]).fillna(resumo["nivel_10d"])
        
        for k in range(OCORRENCIAS_MESMO_DIA):
            col_nome = f"mesmo_dia_ha_{k + 1}_ocorrencias"
            g[col_nome] = [
                resumo["ocorrencias"].get(d, [])[k] if len(resumo["ocorrencias"].get(d, [])) > k else float("nan")
                for d in g["dia_semana"]
            ]
            
        g["nivel_bloco_atual"] = g["bloco"].map(resumo["nivel_bloco"])
        g["dias_observados_no_bloco"] = g["bloco"].map(resumo["dias_bloco"]).fillna(0)
        g["efeito_proteina"] = g["proteina"].map(resumo["proteina"])
        g["efeito_preparo"] = g["preparo"].map(resumo["preparo"])
        g["efeito_prato"] = g["prato_padronizado"].map(resumo["prato_padronizado"])
        g["efeito_evento"] = g["evento_tipo"].map(resumo["evento_tipo"])
        g["efeito_posicao_bloco"] = pd.Series(faixa_do_bloco(g), index=g.index).map(resumo["faixa_bloco"])
        g["mesma_epoca_ano_passado"] = mesma_epoca_ano_passado(g["data"], resumo["serie_razao"])
        
        linhas.append(g)
        
    return pd.concat(linhas, ignore_index=True)


def adicionar_atributos_finais(gold: pd.DataFrame) -> pd.DataFrame:
    """Calcula razões relativas, métricas de antecedência e features de calendário."""
    gold["antecedencia_dias"] = (gold["data"] - gold["origem"]).dt.days
    gold["idade_informacao_dias"] = (gold["origem"] - gold["ultimo_consumo_usado"]).dt.days
    gold["nivel_base"] = (gold["nivel_10d"] + gold["nivel_mesmo_dia"]) / 2.0
    gold["tendencia_nivel"] = gold["nivel_10d"] / gold["nivel_20d"]
    gold["coef_variacao_10d"] = gold["desvio_10d"] / gold["nivel_base"]
    
    for col in RELATIVOS_AO_NIVEL:
        gold[f"rel_{col}"] = gold[col] / gold["nivel_base"]
        
    gold["semana_do_bloco"] = gold["dias_desde_inicio_bloco"] // 5
    gold["evento_especial"] = ~gold["evento_tipo"].isin(EVENTOS_ROTINEIROS)
    
    pausa_curta = gold["pausa_apos_bloco"].lt(PAUSA_FIM_DE_SEMESTRE)
    gold.loc[pausa_curta, "dias_ate_fim_bloco"] = LIMITE_DISTANCIA_BLOCO
    gold["fim_de_bloco"] = gold["dias_ate_fim_bloco"].le(3)
    gold["inicio_de_bloco"] = gold["dias_desde_inicio_bloco"].le(4)
    
    for col in ("dias_ate_fim_bloco", "dias_desde_inicio_bloco"):
        gold[col] = gold[col].clip(upper=LIMITE_DISTANCIA_BLOCO)
        
    for dia, nome in enumerate(["seg", "ter", "qua", "qui", "sex"]):
        gold[f"dia_{nome}"] = gold["dia_semana"].eq(dia)
        
    gold["sem_cardapio"] = gold["proteina"].isna()
    gold["alvo_relativo"] = gold["consumo"] / gold["nivel_base"]
    
    # Oráculo: clima observado real (usado apenas para testes de sensibilidade ex-post)
    rename_dict = {
        "chuva_mm": "oraculo_chuva_mm",
        "chuva_horas": "oraculo_chuva_horas",
        "temp_max": "oraculo_temp_max",
        "vento_max": "oraculo_vento_max"
    }
    return gold.rename(columns=rename_dict)


def construir_janela_horizonte(fato: pd.DataFrame, servidos: pd.DataFrame, dias: pd.DataFrame, nome: str) -> pd.DataFrame:
    """Constrói o dataset Gold para um determinado horizonte (T+1 ou T+2)."""
    deslocamento = HORIZONTES[nome]["deslocamento_dias"]
    alvos = preparar_alvos(fato)
    gold = adicionar_atributos_finais(montar_janela_temporal(alvos, servidos, deslocamento))
    
    gold = gold.merge(dias[["data", "tratamento", "usar_no_treino", "tipo_sugerido"]], on="data", how="left")
    gold["usar_no_treino"] = gold["usar_no_treino"].astype("boolean").fillna(False).astype(bool)
    gold["tratamento"] = gold["tratamento"].fillna("sem_consumo")
    gold["horizonte"] = nome
    
    # Acoplar previsões meteorológicas arquivadas
    caminho_previsto = SILVER / "silver_clima_previsto.parquet"
    if caminho_previsto.exists():
        previsto = pd.read_parquet(caminho_previsto)
        prev_horiz = previsto[previsto["horizonte"] == nome].drop(columns="horizonte")
        gold = gold.merge(prev_horiz, on=["origem", "data"], how="left")
        
    colunas_ordenadas = [
        "origem", "data", "horizonte", "antecedencia_dias", "ano", "mes", "consumo", "alvo_relativo",
        "tratamento", "usar_no_treino", "tipo_sugerido",
        "nivel_base", "nivel_10d", "nivel_20d", "nivel_mesmo_dia", "nivel_exponencial", "nivel_bloco_atual",
        "dias_observados_no_bloco", "ultimo_consumo", "media_5d", "mediana_5d", "media_10d", "media_20d",
        "desvio_5d", "desvio_10d", "minimo_5d", "maximo_5d",
        "mesmo_dia_ha_1_ocorrencias", "mesmo_dia_ha_2_ocorrencias",
        "mesmo_dia_ha_3_ocorrencias", "mesmo_dia_ha_4_ocorrencias",
        "tendencia_nivel", "dispersao_recente", "coef_variacao_10d",
        *[f"rel_{c}" for c in RELATIVOS_AO_NIVEL],
        "idade_informacao_dias", "ultimo_consumo_usado",
        "dia_semana", "dia_seg", "dia_ter", "dia_qua", "dia_qui", "dia_sex",
        "dias_ate_fim_bloco", "dias_desde_inicio_bloco", "semana_do_bloco", "fim_de_bloco", "inicio_de_bloco",
        "vespera_de_pausa", "pos_pausa", "dias_sem_aula_antes", "dias_letivos_na_semana",
        "evento_tipo", "evento_especial", "efeito_evento", "efeito_posicao_bloco", "mesma_epoca_ano_passado",
        "proteina", "preparo", "prato_padronizado", "sem_cardapio", "efeito_proteina", "efeito_preparo", "efeito_prato",
        "previsao_chuva_mm", "previsao_chuva_manha_mm", "previsao_chuva_acum3d_mm", "previsao_temp_max",
        "oraculo_chuva_mm", "oraculo_chuva_horas", "oraculo_temp_max", "oraculo_vento_max"
    ]
    
    # Plano humano apenas em T+1 (Preparo da semana seguinte)
    if nome == "t1":
        colunas_ordenadas += ["reserva_discente", "plano_nutricionista"]
        
    colunas_finais = [c for c in colunas_ordenadas if c in gold.columns]
    return gold[colunas_finais].sort_values(["origem", "data"]).reset_index(drop=True)


def construir_gold(fato: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Constrói ambos os horizontes Gold (t1 e t2) a partir da tabela fato Silver."""
    servidos, candidatos = detectar_dias_atipicos(fato)
    dias = classificar_dias(servidos, candidatos)
    return {nome: construir_janela_horizonte(fato, servidos, dias, nome) for nome in ("t1", "t2")}


DICIONARIO_ATRIBUTOS = [
    ("identificador", "origem", "t1 e t2", "Sexta-feira de emissão da previsão (marco de corte temporal)"),
    ("identificador", "data", "t1 e t2", "Dia previsto para o almoço no refeitório"),
    ("alvo", "consumo", "resultado", "Refeições efetivamente consumidas na catraca"),
    ("alvo", "alvo_relativo", "resultado", "Consumo normalizado em relação ao nível basal (consumo / nivel_base)"),
    ("controle", "tratamento", "resultado", "normal, previsivel_calendario, alto_mantido, isolado_sem_explicacao, etc."),
    ("controle", "usar_no_treino", "resultado", "Booleano indicando elegibilidade para treinamento supervisionado"),
    ("nivel_basal", "nivel_base", "t1 e t2", "Média entre nível_10d e nível_mesmo_dia"),
    ("nivel_basal", "nivel_10d", "t1 e t2", "Mediana dos 10 dias úteis normais anteriores à emissão"),
    ("nivel_basal", "nivel_20d", "t1 e t2", "Mediana dos 20 dias úteis normais anteriores à emissão"),
    ("nivel_basal", "nivel_mesmo_dia", "t1 e t2", "Mediana das 4 ocorrências normais do mesmo dia da semana"),
    ("nivel_basal", "nivel_exponencial", "t1 e t2", "Média móvel exponencial causal com alfa 0.2"),
    ("calendario", "dias_ate_fim_bloco", "t1 e t2", "Dias letivos restantes até o término do semestre acadêmico"),
    ("calendario", "dias_desde_inicio_bloco", "t1 e t2", "Dias letivos transcorridos desde o início do período"),
    ("cardapio", "prato_padronizado", "t1 e t2", "Nome padronizado do prato principal"),
    ("cardapio", "efeito_proteina", "t1 e t2", "Razão histórica de consumo suavizada por prior Bayesiano"),
    ("plano_humano", "plano_nutricionista", "somente t1", "Estimativa operacional lançada pela nutricionista na sexta-feira"),
    ("plano_humano", "reserva_discente", "somente t1", "Reservas discentes ativas no período 2023-2025"),
    ("clima_previsto", "previsao_chuva_mm", "t1 e t2", "Previsão de precipitação arquivada pela API no dia da emissão"),
    ("oraculo_clima", "oraculo_chuva_mm", "ex-post", "Chuva observada na estação meteorológica (apenas para auditoria)")
]


def executar_pipeline_gold() -> dict:
    """Executa a engenharia de atributos Gold e grava os artefatos em disco."""
    GOLD.mkdir(parents=True, exist_ok=True)
    fato = pd.read_parquet(SILVER / "silver_fato_diaria.parquet")
    
    # Salvar dicionário de atributos
    pd.DataFrame(DICIONARIO_ATRIBUTOS, columns=["grupo", "coluna", "disponivel_em", "descricao"]).to_csv(
        GOLD / "gold_dicionario_atributos.csv", sep=";", index=False, encoding="utf-8-sig"
    )
    
    tabelas_gold = construir_gold(fato)
    relatorio = {}
    
    for nome, tabela in tabelas_gold.items():
        caminho_parquet = GOLD / f"gold_{nome}.parquet"
        caminho_csv = GOLD / f"gold_{nome}.csv"
        tabela.to_parquet(caminho_parquet, index=False)
        tabela.to_csv(caminho_csv, sep=";", index=False, encoding="utf-8-sig")
        
        relatorio[nome] = {
            "linhas": len(tabela),
            "colunas": tabela.shape[1],
            "primeira_emissao": tabela["origem"].min().strftime("%d/%m/%Y"),
            "ultima_emissao": tabela["origem"].max().strftime("%d/%m/%Y"),
            "dias_com_consumo": int(tabela["consumo"].notna().sum()),
            "dias_elegiveis_treino": int(tabela["usar_no_treino"].sum())
        }
        
    return relatorio


if __name__ == "__main__":
    relatorio = executar_pipeline_gold()
    print("=== Camada Gold Construída com Sucesso! ===")
    for horizonte, info in relatorio.items():
        print(f"\nHorizonte [{horizonte}]:")
        for k, v in info.items():
            print(f"  {k}: {v}")

