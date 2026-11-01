"""Pipeline de ingestão e transformação das Camadas Bronze para Silver."""
import unicodedata
from pathlib import Path
import pandas as pd

from .config import (
    ANO_INICIO_NUTRICIONISTA,
    BRONZE,
    PAUSA_MINIMA_DIAS_UTEIS_BLOCO,
    SILVER,
)
from .atipicos import detectar_dias_atipicos, classificar_dias

HORAS_MINIMAS_DIA = 20
HORAS_MANHA = (6, 13)
HORAS_MINIMAS_MANHA = 7
DIAS_ACUMULO_CHUVA = 3
EVENTOS_GREVE = ("greve", "atividade_durante_greve")
SIGLAS_DIAS_SEMANA = ["seg", "ter", "qua", "qui", "sex", "sab", "dom"]


def ler_csv_flexivel(caminho: Path) -> pd.DataFrame:
    """Lê CSV tentando UTF-8 com fallback para Latin-1."""
    try:
        return pd.read_csv(caminho, sep=";", encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(caminho, sep=";", encoding="latin1")


def ler_pasta_bronze(subpasta: str) -> pd.DataFrame:
    """Lê e concatena todos os arquivos CSV de uma subpasta em Bronze."""
    pasta = BRONZE / subpasta
    arquivos = sorted(pasta.glob("*.csv"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {pasta}")
    return pd.concat([ler_csv_flexivel(f) for f in arquivos], ignore_index=True)


def padronizar_texto(texto: str) -> str:
    """Remove acentos, pontuações desnecessárias e padroniza em maiúsculas."""
    if pd.isna(texto):
        return texto
    sem_acento = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return " ".join(sem_acento.upper().replace("-", " ").split())


def processar_calendario() -> pd.DataFrame:
    """Processa o calendário escolar com eventos acadêmicos e status legal."""
    cal = ler_pasta_bronze("calendario")
    cal["data"] = pd.to_datetime(cal["data"], dayfirst=True)
    assert not cal["data"].duplicated().any(), "Datas duplicadas no calendário!"
    return cal[["data", "status_legal", "evento_tipo"]].sort_values("data").reset_index(drop=True)


def processar_consumo() -> pd.DataFrame:
    """Processa a série histórica de refeições consumidas na catraca."""
    con = ler_pasta_bronze("consumo")
    con["data"] = pd.to_datetime(con["data"], dayfirst=True)
    con = con[con["consumo"].notna()]
    assert not con["data"].duplicated().any(), "Datas duplicadas no consumo!"
    assert (con["consumo"] >= 0).all(), "Consumo negativo detectado!"
    con = con.rename(columns={"status_dado": "status_consumo", "observacao": "obs_consumo"})
    return con[["data", "consumo", "status_consumo", "obs_consumo"]].sort_values("data").reset_index(drop=True)


def processar_planos() -> pd.DataFrame:
    """Processa as reservas discentes (2023-2025) e estimativas da nutricionista (2026+)."""
    res = ler_pasta_bronze("reservas")
    res["data"] = pd.to_datetime(res["data"], dayfirst=True)
    res = res[res["reservas"].notna()]
    assert not res["data"].duplicated().any(), "Datas duplicadas em reservas/planos!"
    
    eh_discente = res["data"].dt.year < ANO_INICIO_NUTRICIONISTA
    res["reserva_discente"] = res["reservas"].where(eh_discente)
    res["plano_nutricionista"] = res["reservas"].where(~eh_discente)
    return res[["data", "reserva_discente", "plano_nutricionista"]].sort_values("data").reset_index(drop=True)


def processar_cardapio() -> pd.DataFrame:
    """Processa cardápios semanais com normalização de pratos e proteínas."""
    car = ler_pasta_bronze("cardapio")
    car.columns = [c.lower() for c in car.columns]
    car["data"] = pd.to_datetime(car["data"], dayfirst=True)
    assert not car["data"].duplicated().any(), "Datas duplicadas no cardápio!"
    
    car = car.rename(columns={
        "prato_principal": "prato",
        "tipo_proteina_principal": "proteina",
        "tipo_preparo_principal": "preparo",
        "opcao_vegetariana": "vegetariana",
    })
    car["prato_padronizado"] = car["prato"].map(padronizar_texto)
    colunas = ["data", "prato", "prato_padronizado", "proteina", "preparo", "vegetariana", "guarnicao", "sobremesa"]
    return car[colunas].sort_values("data").reset_index(drop=True)


def processar_clima_historico() -> pd.DataFrame:
    """Processa o histórico meteorológico observado."""
    arq_clima = BRONZE / "clima" / "clima_historico.csv"
    if not arq_clima.exists():
        arq_clima = BRONZE / "clima" / "clima_historico_itaquaquecetuba.csv"
    cli = ler_csv_flexivel(arq_clima)
    cli["data"] = pd.to_datetime(cli["data_iso"])
    cols = {
        "precipitation_sum": "chuva_mm",
        "precipitation_hours": "chuva_horas",
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "wind_speed_10m_max": "vento_max",
    }
    return cli.rename(columns=cols)[["data", *cols.values()]].sort_values("data").reset_index(drop=True)


def calcular_sexta_emissao(datas: pd.Series, deslocamento_dias: int) -> pd.Series:
    """Determina a sexta-feira anterior que serviu de marco de emissão da previsão."""
    return datas - pd.to_timedelta(datas.dt.dayofweek + deslocamento_dias, unit="D")


def processar_previsao_clima_t1() -> pd.DataFrame:
    """Processa previsões meteorológicas arquivadas para o horizonte T+1 (Preparo)."""
    caminho = BRONZE / "clima_previsoes" / "previous_runs_horario.csv"
    if not caminho.exists():
        return pd.DataFrame()
        
    hor = pd.read_csv(caminho, sep=";", parse_dates=["time"])
    hor["data"] = hor["time"].dt.normalize()
    colunas_chuva = [f"precipitation_previous_day{d}" for d in range(1, 8)]
    
    chuva = hor.groupby("data")[colunas_chuva].sum(min_count=HORAS_MINIMAS_DIA)
    manha = hor[hor["time"].dt.hour.between(*HORAS_MANHA)].groupby("data")[colunas_chuva].sum(min_count=HORAS_MINIMAS_MANHA)
    acumulada = chuva.rolling(DIAS_ACUMULO_CHUVA, min_periods=DIAS_ACUMULO_CHUVA).sum()
    temp = hor.groupby("data")[[f"temperature_2m_previous_day{d}" for d in range(1, 8)]].max()
    
    tab = pd.DataFrame({"data": chuva.index})
    tab = tab[tab["data"].dt.dayofweek < 5].reset_index(drop=True)
    tab["origem"] = calcular_sexta_emissao(tab["data"], 3)
    dias = ((tab["data"] - tab["origem"]).dt.days + 1).clip(upper=7)
    
    tab["previsao_chuva_mm"] = [chuva.at[d, f"precipitation_previous_day{n}"] for d, n in zip(tab["data"], dias)]
    tab["previsao_chuva_manha_mm"] = [manha.at[d, f"precipitation_previous_day{n}"] for d, n in zip(tab["data"], dias)]
    tab["previsao_chuva_acum3d_mm"] = [acumulada.at[d, f"precipitation_previous_day{n}"] for d, n in zip(tab["data"], dias)]
    tab["previsao_temp_max"] = [temp.at[d, f"temperature_2m_previous_day{n}"] for d, n in zip(tab["data"], dias)]
    
    return tab.assign(horizonte="t1").dropna(subset=["previsao_chuva_mm"])


def processar_previsao_clima_t2() -> pd.DataFrame:
    """Processa previsões de médio prazo arquivadas para o horizonte T+2 (Compras)."""
    caminho = BRONZE / "clima_previsoes" / "single_runs_horario.csv"
    if not caminho.exists():
        return pd.DataFrame()
        
    hor = pd.read_csv(caminho, sep=";", parse_dates=["time", "rodada"])
    hor["data"] = hor["time"].dt.normalize()
    grupos = hor.groupby(["rodada", "modelo", "data"])
    
    dia = pd.DataFrame({
        "chuva": grupos["precipitation"].sum(min_count=HORAS_MINIMAS_DIA),
        "temp": grupos["temperature_2m"].max()
    })
    dia["manha"] = hor[hor["time"].dt.hour.between(*HORAS_MANHA)].groupby(["rodada", "modelo", "data"])["precipitation"].sum(min_count=HORAS_MINIMAS_MANHA)
    dia = dia.reset_index().sort_values(["rodada", "modelo", "data"])
    dia["acum"] = dia.groupby(["rodada", "modelo"])["chuva"].transform(lambda s: s.rolling(DIAS_ACUMULO_CHUVA, min_periods=DIAS_ACUMULO_CHUVA).sum())
    
    tab = dia.groupby(["rodada", "data"], as_index=False).agg(
        previsao_chuva_mm=("chuva", "mean"),
        previsao_chuva_manha_mm=("manha", "mean"),
        previsao_chuva_acum3d_mm=("acum", "mean"),
        previsao_temp_max=("temp", "mean")
    )
    tab = tab[(tab["data"].dt.dayofweek < 5) & (calcular_sexta_emissao(tab["data"], 10) == tab["rodada"])]
    return tab.rename(columns={"rodada": "origem"}).assign(horizonte="t2").dropna(subset=["previsao_chuva_mm"])


def marcar_blocos_letivos(fato: pd.DataFrame) -> pd.DataFrame:
    """Mapeia a posição ordinal de cada dia útil dentro de blocos contínuos de aula.
    
    Permite aos modelos capturarem desmobilização e curvas de frequência discente
    no início e encerramento de períodos acadêmicos.
    """
    util = fato[fato["dia_semana"] < 5].copy()
    util["letivo"] = util["status_legal"].eq("letivo")
    pausas = (~util["letivo"]).groupby(util["letivo"].cumsum()).transform("sum")
    pausa_longa = ~util["letivo"] & (pausas >= PAUSA_MINIMA_DIAS_UTEIS_BLOCO)
    bloco = pausa_longa.astype(int).diff().fillna(0).eq(-1).cumsum()
    
    letivos = util[util["letivo"]].assign(bloco=bloco)
    grupo = letivos.groupby("bloco")["data"]
    letivos["dias_desde_inicio_bloco"] = grupo.cumcount()
    letivos["dias_ate_fim_bloco"] = grupo.cumcount(ascending=False)
    
    return fato.merge(letivos[["data", "bloco", "dias_desde_inicio_bloco", "dias_ate_fim_bloco"]], on="data", how="left")


def integrar_fato_diaria(
    calendario: pd.DataFrame,
    consumo: pd.DataFrame,
    planos: pd.DataFrame,
    cardapio: pd.DataFrame,
    clima: pd.DataFrame
) -> pd.DataFrame:
    """Integra todas as fontes na tabela fato diária padronizada."""
    fato = calendario
    for tabela in (consumo, planos, cardapio, clima):
        fato = fato.merge(tabela, on="data", how="left")
    assert not fato["data"].duplicated().any(), "Chave temporal duplicada na tabela fato!"
    
    fato["ano"] = fato["data"].dt.year
    fato["mes"] = fato["data"].dt.month
    fato["dia_semana"] = fato["data"].dt.dayofweek
    fato["dia_nome"] = fato["dia_semana"].map(dict(enumerate(SIGLAS_DIAS_SEMANA)))
    fato["regime_plano"] = pd.Series("reserva_discente", index=fato.index).where(
        fato["ano"] < ANO_INICIO_NUTRICIONISTA, "nutricionista"
    )
    fato["no_escopo"] = fato["dia_semana"] < 5
    fato["tem_consumo"] = fato["consumo"].notna()
    fato["tem_cardapio"] = fato["prato"].notna()
    
    # Identificação determinística de períodos de greve
    datas_greve = fato.loc[fato["evento_tipo"].isin(EVENTOS_GREVE), "data"]
    fato["periodo_greve"] = fato["data"].between(datas_greve.min(), datas_greve.max()) if not datas_greve.empty else False
    
    return marcar_blocos_letivos(fato)


def salvar_tabela_silver(df: pd.DataFrame, nome: str) -> None:
    """Salva a tabela tratada em formato Parquet (eficiência) e CSV (auditabilidade)."""
    caminho_parquet = SILVER / f"{nome}.parquet"
    caminho_csv = SILVER / f"{nome}.csv"
    df.to_parquet(caminho_parquet, index=False)
    df.to_csv(caminho_csv, sep=";", index=False, encoding="utf-8-sig")


def executar_pipeline_silver() -> dict:
    """Orquestra a execução completa da camada Silver e gera relatórios de consistência."""
    SILVER.mkdir(parents=True, exist_ok=True)
    
    calendario = processar_calendario()
    consumo = processar_consumo()
    planos = processar_planos()
    cardapio = processar_cardapio()
    clima = processar_clima_historico()
    
    fato = integrar_fato_diaria(calendario, consumo, planos, cardapio, clima)
    
    # Previsões meteorológicas horárias arquivadas
    prev_t1 = processar_previsao_clima_t1()
    prev_t2 = processar_previsao_clima_t2()
    previsoes = pd.concat([prev_t1, prev_t2], ignore_index=True) if not prev_t1.empty or not prev_t2.empty else pd.DataFrame()
    
    # Classificação de Atípicos
    servidos, candidatos = detectar_dias_atipicos(fato)
    dias_classificados = classificar_dias(servidos, candidatos)
    
    # Salvar todas as tabelas
    tabelas_salvar = [
        (calendario, "silver_calendario"),
        (consumo, "silver_consumo"),
        (planos, "silver_planos"),
        (cardapio, "silver_cardapio"),
        (clima, "silver_clima"),
        (fato, "silver_fato_diaria"),
        (previsoes, "silver_clima_previsto"),
        (candidatos, "silver_candidatos_atipicos"),
        (dias_classificados, "silver_dias_classificados")
    ]
    for df, nome in tabelas_salvar:
        salvar_tabela_silver(df, nome)
        
    servidos[["data", "referencia", "razao_referencia"]].to_parquet(SILVER / "silver_referencia_consumo.parquet", index=False)
    
    relatorio = {
        "total_dias_calendario": len(fato),
        "dias_com_consumo": int(fato["tem_consumo"].sum()),
        "dias_letivos_com_consumo": int((fato["tem_consumo"] & fato["status_legal"].eq("letivo")).sum()),
        "candidatos_atipicos": len(candidatos),
        "dias_treinamento_elegiveis": int(dias_classificados["usar_no_treino"].sum()),
        "periodo_coberto": f"{fato['data'].min().strftime('%d/%m/%Y')} a {fato['data'].max().strftime('%d/%m/%Y')}"
    }
    return relatorio


if __name__ == "__main__":
    relatorio = executar_pipeline_silver()
    print("=== Pipeline Silver Executado com Sucesso! ===")
    for k, v in relatorio.items():
        print(f"  {k}: {v}")

