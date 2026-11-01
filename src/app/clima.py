"""
Módulo de Integração Meteorológica em Tempo Real
Busca a previsão do tempo para os próximos 14 dias nas coordenadas do campus
utilizando a API pública Open-Meteo, com fallback seguro.
"""

import requests
import pandas as pd
import numpy as np

try:
    import streamlit as st
    cache_clima = st.cache_data(ttl=3600)
except Exception:
    def cache_clima(fn):
        return fn

# Coordenadas geográficas da instituição pública de ensino
LATITUDE = -23.4862
LONGITUDE = -46.3483

@cache_clima
def obter_previsao_clima_futura(dias_frente=14):
    """
    Obtém previsões de chuva acumulada (mm) e temperatura máxima (°C) para os próximos N dias.
    Retorna dataframe com colunas: ['data_iso', 'precipitation_sum', 'temperature_2m_max', 'wind_gusts_10m_max']
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "daily": ["precipitation_sum", "temperature_2m_max", "wind_gusts_10m_max"],
        "timezone": "America/Sao_Paulo",
        "forecast_days": min(dias_frente, 16)
    }
    
    try:
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            daily = data.get("daily", {})
            df_clima = pd.DataFrame({
                "data_iso": daily.get("time", []),
                "precipitation_sum": daily.get("precipitation_sum", []),
                "temperature_2m_max": daily.get("temperature_2m_max", []),
                "wind_gusts_10m_max": daily.get("wind_gusts_10m_max", [])
            })
            # Trata nulos eventuais
            df_clima["precipitation_sum"] = df_clima["precipitation_sum"].fillna(0.0)
            df_clima["temperature_2m_max"] = df_clima["temperature_2m_max"].fillna(24.5)
            df_clima["wind_gusts_10m_max"] = df_clima["wind_gusts_10m_max"].fillna(20.0)
            return df_clima
    except Exception as e:
        print(f"[AVISO] Não foi possível consultar API Open-Meteo ({e}). Utilizando médias sazonais.")
    
    # Fallback seguro com médias históricas do campus
    hoje = pd.Timestamp.now()
    datas_futuras = [(hoje + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(dias_frente)]
    return pd.DataFrame({
        "data_iso": datas_futuras,
        "precipitation_sum": [2.5] * dias_frente,
        "temperature_2m_max": [24.0] * dias_frente,
        "wind_gusts_10m_max": [18.0] * dias_frente
    })

@cache_clima
def obter_clima_data(data_iso):
    """
    Obtém a meteorologia real para uma data específica (passada ou futura) via Open-Meteo API.
    Retorna: (precipitation_sum, temperature_2m_max, fonte_info) ou (None, None, None) se offline.
    """
    dt = pd.to_datetime(data_iso)
    hoje = pd.Timestamp.now().normalize()
    alvo = dt.normalize()
    diff_dias = (alvo - hoje).days

    # 1. Se estiver na janela recente ou futura (-90 dias a +16 dias), usa forecast API
    if -90 <= diff_dias <= 16:
        try:
            url_forecast = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "daily": ["precipitation_sum", "temperature_2m_max"],
                "timezone": "America/Sao_Paulo",
                "start_date": data_iso,
                "end_date": data_iso
            }
            resp = requests.get(url_forecast, params=params, timeout=5)
            if resp.status_code == 200:
                d = resp.json().get("daily", {})
                precip_list = d.get("precipitation_sum", [])
                temp_list = d.get("temperature_2m_max", [])
                if precip_list and temp_list and precip_list[0] is not None and temp_list[0] is not None:
                    return float(precip_list[0]), float(temp_list[0]), "Open-Meteo em Tempo Real"
        except Exception:
            pass

    # 2. Se for data histórica mais antiga (arquivo meteorológico)
    if diff_dias < 0:
        try:
            url_archive = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "daily": ["precipitation_sum", "temperature_2m_max"],
                "timezone": "America/Sao_Paulo",
                "start_date": data_iso,
                "end_date": data_iso
            }
            resp2 = requests.get(url_archive, params=params, timeout=5)
            if resp2.status_code == 200:
                d2 = resp2.json().get("daily", {})
                precip_list = d2.get("precipitation_sum", [])
                temp_list = d2.get("temperature_2m_max", [])
                if precip_list and temp_list and precip_list[0] is not None and temp_list[0] is not None:
                    return float(precip_list[0]), float(temp_list[0]), "Open-Meteo Arquivo Histórico"
        except Exception:
            pass

    return None, None, "Indisponível"
