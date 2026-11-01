"""
Módulo de Armazenamento e Gestão de Dados Operacionais
Garante que os arquivos históricos e notebooks permaneçam 100% imutáveis.
Armazena novos lançamentos e edições operacionais em 'dados/producao/'.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRODUCAO_DIR = os.path.join(ROOT_DIR, "dados", "producao")
SILVER_CONSUMO = os.path.join(ROOT_DIR, "dados", "silver", "silver_consumo.parquet")
SILVER_FATO = os.path.join(ROOT_DIR, "dados", "silver", "silver_fato_diaria.parquet")
NOVOS_CONSUMOS = os.path.join(PRODUCAO_DIR, "novos_consumos.parquet")
EDICOES_OPERACIONAIS = os.path.join(PRODUCAO_DIR, "edicoes_operacionais.parquet")
PLANEJAMENTO_NUTRI = os.path.join(PRODUCAO_DIR, "planejamento_nutri.parquet")
PREDICOES_HISTORICO = os.path.join(PRODUCAO_DIR, "historico_predicoes.parquet")
CSV_CONSUMO_ATUALIZADO = os.path.join(PRODUCAO_DIR, "consumo_diario_atualizado.csv")

os.makedirs(PRODUCAO_DIR, exist_ok=True)

def carregar_edicoes_operacionais():
    """Carrega o registro de alterações operacionais efetuadas pelo usuário."""
    if os.path.exists(EDICOES_OPERACIONAIS):
        return pd.read_parquet(EDICOES_OPERACIONAIS)
    return pd.DataFrame()

def carregar_fato_diaria_completa():
    """
    Carrega a tabela fato diária unificada (Silver) e sobrepõe as edições
    operacionais salvas em produção, garantindo dados sempre atualizados.
    """
    if os.path.exists(SILVER_FATO):
        df_fato = pd.read_parquet(SILVER_FATO).copy()
    else:
        df_fato = pd.DataFrame()

    # Normalização de Nomes e Compatibilidade com o Schema Atualizado
    if not df_fato.empty:
        if "data" in df_fato.columns and "data_iso" not in df_fato.columns:
            df_fato["data_dt"] = pd.to_datetime(df_fato["data"])
            df_fato["data_iso"] = df_fato["data_dt"].dt.strftime("%Y-%m-%d")
        elif "data_iso" in df_fato.columns and "data_dt" not in df_fato.columns:
            df_fato["data_dt"] = pd.to_datetime(df_fato["data_iso"])
            df_fato["data"] = df_fato["data_dt"]
            
        if "status_legal" in df_fato.columns and "is_dia_letivo" not in df_fato.columns:
            df_fato["is_dia_letivo"] = df_fato["status_legal"].astype(str).str.lower().str.contains("letivo|aula")
            
        if "dia_nome" in df_fato.columns and "dia_da_semana" not in df_fato.columns:
            df_fato["dia_da_semana"] = df_fato["dia_nome"].astype(str).str.lower()
            
        if "prato" in df_fato.columns and "prato_principal" not in df_fato.columns:
            df_fato["prato_principal"] = df_fato["prato"]
            
        if "proteina" in df_fato.columns and "proteina_tipo" not in df_fato.columns:
            df_fato["proteina_tipo"] = df_fato["proteina"]
            
        if "preparo" in df_fato.columns and "preparo_tipo" not in df_fato.columns:
            df_fato["preparo_tipo"] = df_fato["preparo"]
            
        if "plano_nutricionista" in df_fato.columns and "planejado_nutri" not in df_fato.columns:
            df_fato["planejado_nutri"] = df_fato["plano_nutricionista"]
            
        if "reserva_discente" in df_fato.columns and "reservas" not in df_fato.columns:
            df_fato["reservas"] = df_fato["reserva_discente"]
            
        if "chuva_mm" in df_fato.columns and "precipitation_sum" not in df_fato.columns:
            df_fato["precipitation_sum"] = df_fato["chuva_mm"]
            
        if "temp_max" in df_fato.columns and "temperature_2m_max" not in df_fato.columns:
            df_fato["temperature_2m_max"] = df_fato["temp_max"]
            
        if "vento_max" in df_fato.columns and "wind_speed_10m_max" not in df_fato.columns:
            df_fato["wind_speed_10m_max"] = df_fato["vento_max"]

    # Garantir que a coluna 'planejado_nutri' sempre exista na base fato
    if "planejado_nutri" not in df_fato.columns:
        df_fato["planejado_nutri"] = np.nan

    # No regime_reserva == 'nutricionista_estimativa' ou 'nutricionista',
    # a quantidade estipulada pela nutricionista consta no registro de reservas.
    if "regime_reserva" in df_fato.columns and "reservas" in df_fato.columns:
        mask_nutri = df_fato["regime_reserva"].isin(["nutricionista_estimativa", "nutricionista"])
        df_fato.loc[mask_nutri & df_fato["planejado_nutri"].isna(), "planejado_nutri"] = df_fato.loc[mask_nutri, "reservas"]

    # Sobrepor planejamento inserido pela nutricionista em producao
    if os.path.exists(PLANEJAMENTO_NUTRI):
        df_nutri = pd.read_parquet(PLANEJAMENTO_NUTRI)
        if not df_nutri.empty and "data_iso" in df_nutri.columns:
            for _, row_n in df_nutri.iterrows():
                idx_n = df_fato[df_fato["data_iso"] == row_n["data_iso"]].index
                if len(idx_n) > 0:
                    if "planejado_nutri" in row_n and pd.notna(row_n["planejado_nutri"]):
                        df_fato.loc[idx_n, "planejado_nutri"] = row_n["planejado_nutri"]
                    if "prato_principal" in row_n and pd.notna(row_n["prato_principal"]) and row_n["prato_principal"]:
                        df_fato.loc[idx_n, "prato_principal"] = row_n["prato_principal"]
                    if "proteina_tipo" in row_n and pd.notna(row_n["proteina_tipo"]) and row_n["proteina_tipo"]:
                        df_fato.loc[idx_n, "proteina_tipo"] = row_n["proteina_tipo"]

    # Sobrepor novos consumos avulsos se houver
    if os.path.exists(NOVOS_CONSUMOS):
        df_novos_c = pd.read_parquet(NOVOS_CONSUMOS)
        for _, row in df_novos_c.iterrows():
            idx = df_fato[df_fato["data_iso"] == row["data_iso"]].index
            if len(idx) > 0:
                df_fato.loc[idx, "consumo"] = row["consumo"]
            else:
                novo_df = pd.DataFrame([{
                    "data_iso": row["data_iso"],
                    "data": row.get("data", ""),
                    "consumo": row["consumo"],
                    "is_dia_letivo": True
                }])
                df_fato = pd.concat([df_fato, novo_df], ignore_index=True)

    # Sobrepor edições operacionais completas
    if os.path.exists(EDICOES_OPERACIONAIS):
        df_edicoes = pd.read_parquet(EDICOES_OPERACIONAIS)
        for _, row in df_edicoes.iterrows():
            d_iso = row["data_iso"]
            idx = df_fato[df_fato["data_iso"] == d_iso].index
            if len(idx) > 0:
                for col in df_edicoes.columns:
                    if col not in ["data_iso", "timestamp_edicao"]:
                        # Se o campo da edição for nulo/vazio, não sobrescreve dado válido existente na base
                        val = row[col]
                        if pd.isna(val) or val is None or val == "":
                            continue
                        df_fato.loc[idx, col] = val
            else:
                novo_df = pd.DataFrame([row.to_dict()])
                df_fato = pd.concat([df_fato, novo_df], ignore_index=True)

    if not df_fato.empty and "data_iso" in df_fato.columns:
        df_fato["data_dt"] = pd.to_datetime(df_fato["data_iso"])
        df_fato = df_fato.sort_values("data_dt").reset_index(drop=True)

    return df_fato

def consultar_dados_completos_data(data_str):
    """
    Retorna dicionário completo com todas as informações cadastradas para uma data específica:
    - Consumo, Reservas, Planejamento da Nutricionista
    - Cardápio (Prato, Proteína, Preparo, Guarnição, Sobremesa)
    - Clima (Chuva mm, Temperatura máx °C, Vento)
    - Calendário (Letivo, Evento, Status legal)
    """
    dt = pd.to_datetime(data_str)
    data_iso = dt.strftime("%Y-%m-%d")
    data_fmt = dt.strftime("%d/%m/%Y")
    dias_pt = {0: "Segunda-feira", 1: "Terça-feira", 2: "Quarta-feira", 3: "Quinta-feira", 4: "Sexta-feira", 5: "Sábado", 6: "Domingo"}
    dia_semana_nome = dias_pt.get(dt.dayofweek, "Dia Útil")

    df_fato = carregar_fato_diaria_completa()
    reg = df_fato[df_fato["data_iso"] == data_iso]

    resultado = {
        "data_iso": data_iso,
        "data_formatada": data_fmt,
        "dia_semana": dia_semana_nome,
        "encontrado_base": not reg.empty,
        "is_dia_letivo": True,
        "evento_tipo": "Letivo Regular",
        "consumo": None,
        "reservas": None,
        "planejado_nutri": None,
        "prato_principal": "",
        "proteina_tipo": "Frango",
        "preparo_tipo": "Assado / Grelhado",
        "guarnicao": "Arroz e Feijão",
        "sobremesa": "Fruta da Estação",
        "precipitation_sum": 0.0,
        "temperature_2m_max": 25.0,
        "wind_speed_10m_max": 10.0,
        "observacao": ""
    }

    if not reg.empty:
        r = reg.iloc[-1]
        resultado["is_dia_letivo"] = bool(r.get("is_dia_letivo", True))
        resultado["evento_tipo"] = str(r.get("evento_tipo", "Letivo Regular")) if pd.notna(r.get("evento_tipo")) else "Letivo Regular"
        resultado["consumo"] = float(r["consumo"]) if pd.notna(r.get("consumo")) else None
        resultado["reservas"] = float(r["reservas"]) if pd.notna(r.get("reservas")) else None
        resultado["planejado_nutri"] = float(r["planejado_nutri"]) if pd.notna(r.get("planejado_nutri")) else (
            float(r["reservas"]) if (pd.notna(r.get("reservas")) and r.get("regime_reserva") in ["nutricionista", "nutricionista_estimativa"]) else None
        )
        resultado["prato_principal"] = str(r.get("prato_principal", "")) if pd.notna(r.get("prato_principal")) else ""
        resultado["proteina_tipo"] = str(r.get("proteina_tipo", "Frango")) if pd.notna(r.get("proteina_tipo")) else "Frango"
        resultado["preparo_tipo"] = str(r.get("preparo_tipo", "Assado / Grelhado")) if pd.notna(r.get("preparo_tipo")) else "Assado / Grelhado"
        resultado["guarnicao"] = str(r.get("guarnicao", "Arroz e Feijão")) if pd.notna(r.get("guarnicao")) else "Arroz e Feijão"
        resultado["sobremesa"] = str(r.get("sobremesa", "Fruta")) if pd.notna(r.get("sobremesa")) else "Fruta"
        resultado["precipitation_sum"] = float(r["precipitation_sum"]) if pd.notna(r.get("precipitation_sum")) else 0.0
        resultado["temperature_2m_max"] = float(r["temperature_2m_max"]) if pd.notna(r.get("temperature_2m_max")) else 25.0
        resultado["wind_speed_10m_max"] = float(r["wind_speed_10m_max"]) if pd.notna(r.get("wind_speed_10m_max")) else 10.0
        resultado["observacao"] = str(r.get("observacao", "")) if pd.notna(r.get("observacao")) else ""
        resultado["clima_fonte"] = "Base Consolidada (Silver)" if pd.notna(r.get("temperature_2m_max")) else "Padrão"

    # Se temperatura ou chuva não estavam consolidadas na base, consulta API Open-Meteo
    if reg.empty or pd.isna(reg.iloc[-1].get("temperature_2m_max")):
        try:
            from .clima import obter_clima_data
            p_api, t_api, fonte = obter_clima_data(data_iso)
            if t_api is not None:
                resultado["temperature_2m_max"] = float(t_api)
                resultado["precipitation_sum"] = float(p_api) if p_api is not None else 0.0
                resultado["clima_fonte"] = fonte
        except Exception:
            pass

    # Se a nutricionista tiver gravado planejamento semanal nessa data, sobrepõe
    df_nutri = carregar_planejamento_nutricionista()
    if not df_nutri.empty:
        reg_n = df_nutri[df_nutri["data_iso"] == data_iso]
        if not reg_n.empty:
            rn = reg_n.iloc[-1]
            if pd.notna(rn.get("planejado_nutri")):
                resultado["planejado_nutri"] = float(rn["planejado_nutri"])
            if pd.notna(rn.get("prato_principal")) and rn.get("prato_principal"):
                resultado["prato_principal"] = str(rn["prato_principal"])
            if pd.notna(rn.get("proteina_tipo")) and rn.get("proteina_tipo"):
                resultado["proteina_tipo"] = str(rn["proteina_tipo"])
            if pd.notna(rn.get("preparo_tipo")) and rn.get("preparo_tipo"):
                resultado["preparo_tipo"] = str(rn["preparo_tipo"])

    return resultado

def salvar_edicao_operacional_completa(data_str, dados_editados):
    """
    Grava ou atualiza todos os atributos operacionais de uma data em 'edicoes_operacionais.parquet'.
    Preserva a base histórica intacta e sincroniza a visualização em tempo real.
    """
    dt = pd.to_datetime(data_str)
    data_iso = dt.strftime("%Y-%m-%d")
    data_fmt = dt.strftime("%d/%m/%Y")

    registro = {
        "data_iso": data_iso,
        "data": data_fmt,
        "data_dt": dt,
        "timestamp_edicao": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    registro.update(dados_editados)

    novo_df = pd.DataFrame([registro])

    if os.path.exists(EDICOES_OPERACIONAIS):
        df_atual = pd.read_parquet(EDICOES_OPERACIONAIS)
        df_final = pd.concat([df_atual, novo_df], ignore_index=True).drop_duplicates(subset=["data_iso"], keep="last")
    else:
        df_final = novo_df

    df_final.to_parquet(EDICOES_OPERACIONAIS, index=False)

    # Se houver edição de consumo, sincroniza também em novos_consumos e CSV
    if "consumo" in dados_editados and dados_editados["consumo"] is not None:
        salvar_consumo_diario(data_iso, dados_editados["consumo"])

    # Se houver planejamento da nutricionista ou cardápio, sincroniza em planejamento_nutri
    if any(k in dados_editados for k in ["planejado_nutri", "prato_principal", "proteina_tipo", "preparo_tipo"]):
        dias_pt = {0: "Segunda-feira", 1: "Terça-feira", 2: "Quarta-feira", 3: "Quinta-feira", 4: "Sexta-feira", 5: "Sábado", 6: "Domingo"}
        df_n = pd.DataFrame([{
            "data_iso": data_iso,
            "data": data_fmt,
            "dia_semana": dias_pt.get(dt.dayofweek, "Dia Útil"),
            "prato_principal": dados_editados.get("prato_principal", ""),
            "proteina_tipo": dados_editados.get("proteina_tipo", "Frango"),
            "preparo_tipo": dados_editados.get("preparo_tipo", "Assado / Grelhado"),
            "planejado_nutri": dados_editados.get("planejado_nutri", None)
        }])
        salvar_planejamento_semanal(df_n)

    return True

def carregar_historico_consumo():
    """Carrega o histórico de consumo consolidado."""
    df_fato = carregar_fato_diaria_completa()
    if "consumo" in df_fato.columns:
        df_c = df_fato[df_fato["consumo"].notna()][["data", "data_iso", "consumo"]].copy()
        df_c["data_dt"] = pd.to_datetime(df_c["data_iso"])
        return df_c.sort_values("data_dt").reset_index(drop=True)
    return pd.DataFrame(columns=["data", "data_iso", "consumo", "data_dt"])

def consultar_consumo_por_data(data_str):
    """Retorna o valor de consumo registrado para a data."""
    dados = consultar_dados_completos_data(data_str)
    return dados.get("consumo")

def salvar_consumo_diario(data_str, consumo_val):
    """Salva o consumo diário em novos_consumos.parquet e CSV consolidado."""
    dt = pd.to_datetime(data_str)
    data_iso = dt.strftime("%Y-%m-%d")
    data_formatada = dt.strftime("%d/%m/%Y")

    novo_reg = pd.DataFrame([{
        "data_dt": dt,
        "data": data_formatada,
        "data_iso": data_iso,
        "consumo": float(consumo_val)
    }])

    if os.path.exists(NOVOS_CONSUMOS):
        df_novos = pd.read_parquet(NOVOS_CONSUMOS)
        df_novos = pd.concat([df_novos, novo_reg], ignore_index=True).drop_duplicates(subset=["data_iso"], keep="last")
    else:
        df_novos = novo_reg

    df_novos.to_parquet(NOVOS_CONSUMOS, index=False)

    df_completo = carregar_historico_consumo()
    df_completo[["data", "data_iso", "consumo"]].to_csv(CSV_CONSUMO_ATUALIZADO, index=False, encoding="utf-8-sig")
    return True

def carregar_planejamento_nutricionista():
    """Retorna o planejamento inserido pela nutricionista."""
    if os.path.exists(PLANEJAMENTO_NUTRI):
        return pd.read_parquet(PLANEJAMENTO_NUTRI).sort_values("data_iso").reset_index(drop=True)
    return pd.DataFrame(columns=["data_iso", "data", "dia_semana", "prato_principal", "proteina_tipo", "preparo_tipo", "planejado_nutri"])

def salvar_planejamento_semanal(df_semana):
    """Salva a grade de planejamento semanal."""
    if os.path.exists(PLANEJAMENTO_NUTRI):
        df_atual = pd.read_parquet(PLANEJAMENTO_NUTRI)
        df_unificado = pd.concat([df_atual, df_semana], ignore_index=True).drop_duplicates(subset=["data_iso"], keep="last")
    else:
        df_unificado = df_semana
    df_unificado.to_parquet(PLANEJAMENTO_NUTRI, index=False)
    return True

def salvar_predicoes_geradas(df_predicoes):
    """Armazena o histórico das previsões emitidas."""
    try:
        df_predicoes["timestamp_execucao"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if os.path.exists(PREDICOES_HISTORICO):
            df_hist = pd.read_parquet(PREDICOES_HISTORICO)
            df_final = pd.concat([df_hist, df_predicoes], ignore_index=True)
        else:
            df_final = df_predicoes
        df_final.to_parquet(PREDICOES_HISTORICO, index=False)
    except Exception:
        pass
    return True

def listar_ultimas_edicoes(limite=10):
    """Retorna tabela com as últimas edições salvas pelo usuário."""
    if os.path.exists(EDICOES_OPERACIONAIS):
        df = pd.read_parquet(EDICOES_OPERACIONAIS)
        if not df.empty and "timestamp_edicao" in df.columns:
            return df.sort_values("timestamp_edicao", ascending=False).head(limite)
    return pd.DataFrame()
