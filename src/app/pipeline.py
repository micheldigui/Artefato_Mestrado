"""
Módulo do Pipeline Analítico e Inferência Preditiva
Serve os modelos escolhidos em Fev-Ago/2026: melhor modelo no melhor cenário de cada horizonte, congelados
em 31/08/2026, com margens empíricas por quantil calibradas no mesmo período.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from storage import carregar_planejamento_nutricionista, salvar_predicoes_geradas, carregar_fato_diaria_completa
    from clima import obter_previsao_clima_futura, obter_clima_data
    from modelos import (
        FIM_TREINO_CONGELADO, atributos_da_emissao, carregar_gold, carregar_margens, combinar_hibrido, linha_arquivada,
        algoritmos_do_torneio, margem_do_algoritmo, modelo_congelado, modelo_walk_forward, nome_do_modelo, cenario_oficial, origem_da_data, prever
    )
except ImportError:
    from .storage import carregar_planejamento_nutricionista, salvar_predicoes_geradas, carregar_fato_diaria_completa
    from .clima import obter_previsao_clima_futura, obter_clima_data
    from .modelos import (
        FIM_TREINO_CONGELADO, atributos_da_emissao, carregar_gold, carregar_margens, combinar_hibrido, linha_arquivada,
        algoritmos_do_torneio, margem_do_algoritmo, modelo_congelado, modelo_walk_forward, nome_do_modelo, cenario_oficial, origem_da_data, prever
    )

DIAS_SEMANA_PT = {
    0: "Segunda-feira",
    1: "Terça-feira",
    2: "Quarta-feira",
    3: "Quinta-feira",
    4: "Sexta-feira",
    5: "Sábado",
    6: "Domingo"
}

# Mapeamento Oficial de Feriados Nacionais e Recessos Escolares
FERIADOS_FIXOS_BR = {
    "01-01": "Confraternização Universal",
    "04-21": "Tiradentes",
    "05-01": "Dia do Trabalho",
    "07-09": "Revolução Constitucionalista",
    "09-07": "Independência do Brasil",
    "10-12": "Nossa Sra. Aparecida",
    "11-02": "Finados",
    "11-15": "Proclamação da República",
    "11-20": "Dia da Consciência Negra",
    "12-25": "Natal"
}

# Cenários de dados da pesquisa. As métricas estão nas tabelas de resultados/tabelas_dissertacao.
CENARIOS_DICT = {
    "Cenário 1 = Consumo Histórico + Calendário": {
        "numero": 1,
        "usar_clima": False,
        "usar_cardapio": False,
        "usar_nutri": False,
        "descricao": "Base operacional: consumo recente e calendário letivo, sem cardápio, clima ou quantitativo."
    },
    "Cenário 2 = Cenário 1 + Cardápio": {
        "numero": 2,
        "usar_clima": False,
        "usar_cardapio": True,
        "usar_nutri": False,
        "descricao": "Cenário 1 + atratividade histórica do cardápio. É a configuração do modelo congelado de referência."
    },
    "Cenário 3 = Cenário 2 + Clima": {
        "numero": 3,
        "usar_clima": True,
        "usar_cardapio": True,
        "usar_nutri": False,
        "descricao": "Cenário 2 + previsão de chuva e temperatura disponível na emissão."
    },
    "Cenário 4 = Cenário 3 + Quantitativo Institucional": {
        "numero": 4,
        "usar_clima": True,
        "usar_cardapio": True,
        "usar_nutri": True,
        "descricao": "Cenário 3 + quantitativo institucional como atributo. Exclusivo do preparo (T+1)."
    }
}
CENARIO_PADRAO = next(nome for nome, cfg in CENARIOS_DICT.items() if cfg["numero"] == cenario_oficial("t1"))
CENARIO_COMPRAS = next(nome for nome, cfg in CENARIOS_DICT.items() if cfg["numero"] == cenario_oficial("t2"))

NOME_PREPARO, NOME_COMPRAS = nome_do_modelo("t1"), nome_do_modelo("t2")
MODELO_HIBRIDO = f"Modelo Híbrido ({NOME_PREPARO} + Quantitativo Institucional)"
MODELO_IA = "Inteligência Artificial (Modelo Escolhido)"
MODELO_MANUAL = "Planejamento Manual da Gestão (Sem IA)"
MODELOS_RANKING = {
    MODELO_HIBRIDO: {
        "tipo": "Human-in-the-Loop",
        "detalhe": f"Modelo oficial do preparo: combina em partes iguais o {NOME_PREPARO} e o quantitativo informado pela nutricionista"
    },
    MODELO_IA: {
        "tipo": "Modelo congelado em 31/08/2026",
        "detalhe": f"Modelo escolhido sem o quantitativo: {NOME_PREPARO} no preparo e {NOME_COMPRAS} nas compras"
    },
    MODELO_MANUAL: {
        "tipo": "Estimativa Humana da Gestão",
        "detalhe": "Estimativa informada pela nutricionista, sem uso de inteligência artificial"
    }
}

# O simulador oferece todos os algoritmos; as telas de previsão usam só os modelos oficiais
MODELOS_SIMULADOR = {**MODELOS_RANKING}
for _chave, _nome, _tipo in algoritmos_do_torneio():
    MODELOS_SIMULADOR[_nome] = {
        "tipo": _tipo, "algoritmo": _chave,
        "detalhe": f"{_nome}: algoritmo do torneio. Usa o cenário escolhido, com treino até a sexta de corte nas datas até 31/08/2026 e modelo congelado depois"
    }


def opcoes_simulador(objetivo):
    """Lista os modelos do simulador, sempre com o campeão do objetivo em primeiro lugar."""
    algoritmos = [nome for nome, info in MODELOS_SIMULADOR.items() if "algoritmo" in info]
    if objetivo == "compras":
        return [NOME_COMPRAS] + [nome for nome in algoritmos if nome != NOME_COMPRAS]
    return [MODELO_HIBRIDO, MODELO_MANUAL] + algoritmos


def rotulo_simulador(objetivo, modelo):
    campeao = NOME_COMPRAS if objetivo == "compras" else MODELO_HIBRIDO
    return f"{modelo} (campeão)" if modelo == campeao else modelo


_MARGENS = carregar_margens()
MARGEM_PADRAO = "Margem P85 (Padrão da Pesquisa)"
MARGEM_SEM = "Predição Pura (Sem Margem)"
MARGENS_INFO = {
    MARGEM_PADRAO: {
        "quantil": "P85",
        "cobertura": f"{_MARGENS['P85']['cobertura_hibrido']:.1f}% no híbrido em Fev-Ago".replace(".", ","),
        "descricao": f"Política padrão: +{_MARGENS['P85']['hibrido']:.0f} ref no híbrido e +{_MARGENS['P85']['ia']:.0f} ref na IA"
    },
    "Margem P90 (Máxima Proteção)": {
        "quantil": "P90",
        "cobertura": f"{_MARGENS['P90']['cobertura_hibrido']:.1f}% no híbrido em Fev-Ago".replace(".", ","),
        "descricao": f"Conservadora: +{_MARGENS['P90']['hibrido']:.0f} ref no híbrido e +{_MARGENS['P90']['ia']:.0f} ref na IA"
    },
    "Margem P75 (Econômica)": {
        "quantil": "P75",
        "cobertura": f"{_MARGENS['P75']['cobertura_hibrido']:.1f}% no híbrido em Fev-Ago".replace(".", ","),
        "descricao": f"Econômica: +{_MARGENS['P75']['hibrido']:.0f} ref no híbrido e +{_MARGENS['P75']['ia']:.0f} ref na IA"
    },
    MARGEM_SEM: {
        "quantil": None,
        "cobertura": "sem margem",
        "descricao": "Estimativa central, sem acréscimo de segurança"
    }
}


def numero_do_cenario(cenario_escolhido):
    """Localiza o cenário pelo nome completo ou pelo prefixo e devolve sua configuração."""
    prefixo = str(cenario_escolhido).split("=")[0].split(":")[0].split("(")[0].strip()
    for nome, cfg in CENARIOS_DICT.items():
        if nome == cenario_escolhido or nome.split("=")[0].strip() == prefixo:
            return cfg
    return CENARIOS_DICT[CENARIO_PADRAO]


def calcular_margem(tipo_margem, politica, margem_custom=0.0):
    """Devolve a margem da política: 'hibrido', 'ia', 't2' ou 'manual'."""
    if "Personalizada" in str(tipo_margem):
        return float(margem_custom)
    quantil = next((q for q in ("P75", "P85", "P90") if q in str(tipo_margem)), None)
    if quantil is None or politica == "manual":
        return 0.0
    return _MARGENS[quantil][politica]


def margem_do_modelo(tipo_margem, politica, margem_custom, horizonte, cenario, algoritmo):
    """Usa a margem do próprio algoritmo no simulador e a margem oficial nos demais casos."""
    quantil = next((q for q in ("P75", "P85", "P90") if q in str(tipo_margem)), None)
    if algoritmo is None or quantil is None or "Personalizada" in str(tipo_margem):
        return calcular_margem(tipo_margem, politica, margem_custom)
    return margem_do_algoritmo(horizonte, cenario, algoritmo, int(quantil[1:]) / 100)


def cenario_da_ia(cfg, horizonte, eh_hibrido):
    """O híbrido e o T+2 nunca recebem o quantitativo como atributo da IA."""
    return 3 if cfg["numero"] == 4 and (eh_hibrido or horizonte == "t2") else cfg["numero"]


def aplicar_entradas(linha, cenario, val_nutri, chuva=None, temp=None):
    """Acrescenta à linha de atributos as entradas do dia exigidas pelo cenário."""
    linha = linha.copy()
    if cenario >= 3 and chuva is not None:
        linha["previsao_chuva_mm"] = chuva
        linha["previsao_temp_max"] = temp
    if cenario == 4:
        linha["rel_plano_humano"] = (val_nutri / linha["nivel_base"]) if val_nutri else 1.0
    return linha


def obter_proximos_dias_uteis(origem_dt=None, num_dias=10, dias_frente=None):
    """
    Retorna as datas úteis estruturadas rigorosamente por semanas civis letivas (segunda a sexta):
    - Dias 0 a 4: Semana t+1 (Segunda a Sexta da semana seguinte, ex: 02/11/2026 a 06/11/2026)
    - Dias 5 a 9: Semana t+2 (Segunda a Sexta da semana subsequente, ex: 09/11/2026 a 13/11/2026)
    Garante que feriados (como 02/11) permaneçam na grade da sua respectiva semana com previsão 0,
    sem deslocar dias para semanas posteriores.
    """
    if isinstance(origem_dt, (int, float)):
        num_dias = int(origem_dt)
        origem_dt = None

    from datetime import date
    if dias_frente is not None:
        num_dias = dias_frente
        
    import zoneinfo
    try:
        tz_sp = zoneinfo.ZoneInfo("America/Sao_Paulo")
    except Exception:
        tz_sp = None

    if origem_dt is None:
        if tz_sp:
            origem_dt = datetime.now(tz_sp)
        else:
            origem_dt = datetime.utcnow() - timedelta(hours=3)

    if not isinstance(origem_dt, datetime):
        origem_dt = datetime.combine(origem_dt, datetime.min.time())

    # Determinar a Segunda-feira da Semana t+1
    # No corte da pesquisa/artefato com histórico até o fim de outubro/2026:
    # A semana t+1 operacional é de 02/11/2026 a 06/11/2026.
    dt_pura = origem_dt.date() if hasattr(origem_dt, "date") else origem_dt
    if hasattr(dt_pura, "year") and dt_pura.year == 2026 and dt_pura.month in [10, 11] and (dt_pura <= date(2026, 11, 8)):
        segunda_t1 = datetime(2026, 11, 2)
    else:
        dia_semana = origem_dt.weekday()
        if dia_semana in [0, 1, 2, 3]:
            segunda_t1 = origem_dt - timedelta(days=dia_semana)
        elif dia_semana == 4:
            if origem_dt.hour >= 14:
                segunda_t1 = origem_dt + timedelta(days=3)
            else:
                segunda_t1 = origem_dt - timedelta(days=4)
        elif dia_semana == 5:
            segunda_t1 = origem_dt + timedelta(days=2)
        else:
            segunda_t1 = origem_dt + timedelta(days=1)
            
    segunda_t1 = datetime(segunda_t1.year, segunda_t1.month, segunda_t1.day)
    segunda_t2 = segunda_t1 + timedelta(days=7)

    dias_t1 = [segunda_t1 + timedelta(days=i) for i in range(5)]
    dias_t2 = [segunda_t2 + timedelta(days=i) for i in range(5)]
    
    todos_dias = dias_t1 + dias_t2
    return todos_dias[:num_dias]

def executar_pipeline_previsao(
    horizonte_alvo="todos",
    cenario_escolhido=CENARIO_PADRAO,
    modelo_escolhido=MODELO_HIBRIDO,
    tipo_margem=MARGEM_PADRAO,
    margem_adicional=0
):
    """
    Executa a inferência para os horizontes futuros com flexibilidade de seleção de dados
    e retorna tanto a tabela de previsões consolidadas quanto a matriz de features geradas.
    """
    cfg = numero_do_cenario(cenario_escolhido)
    usar_clima = cfg["usar_clima"]
    usar_cardapio = cfg["usar_cardapio"]
    eh_manual = "Manual" in modelo_escolhido
    pede_hibrido = "Híbrido" in modelo_escolhido

    df_planejamento = carregar_planejamento_nutricionista()
    df_fato = carregar_fato_diaria_completa()
    df_clima_futuro = obter_previsao_clima_futura(dias_frente=16)
    mediana_temp = 24.5

    datas_uteis = obter_proximos_dias_uteis(num_dias=10)
    atributos = {
        "t1": atributos_da_emissao(df_fato, origem_da_data(datas_uteis[0], "t1"), "t1"),
        "t2": atributos_da_emissao(df_fato, origem_da_data(datas_uteis[5], "t2"), "t2")
    }

    linhas_pred = []
    linhas_features = []
    
    for idx, dt in enumerate(datas_uteis):
        data_iso = dt.strftime("%Y-%m-%d")
        data_formatada = dt.strftime("%d/%m/%Y")
        dia_sem_num = dt.weekday()
        dia_sem_nome = DIAS_SEMANA_PT[dia_sem_num]
        horizonte = "Semana_t1" if idx < 5 else "Pos_Semana_t2"
        hz = "t1" if idx < 5 else "t2"
        eh_hibrido = pede_hibrido and hz == "t1"
        # O quantitativo institucional não existe para o T+2
        usar_nutri = hz == "t1" and (cfg["usar_nutri"] or eh_hibrido or eh_manual)
        
        if horizonte_alvo != "todos" and horizonte != horizonte_alvo:
            continue
        
        # Identificação de Dia Letivo no Calendário Acadêmico
        row_fato = df_fato[df_fato["data_iso"] == data_iso]
        is_letivo = True
        evento_nome = "Letivo Regular"
        if not row_fato.empty:
            is_letivo = bool(row_fato.iloc[0].get("is_dia_letivo", True))
            evento_nome = str(row_fato.iloc[0].get("evento_tipo", "Letivo Regular"))
            status_leg = str(row_fato.iloc[0].get("status_legal", "")).lower().strip()
            if status_leg == "nao_letivo" or evento_nome.lower() in ["feriado", "recesso", "ferias", "sem_atividade_letiva", "final_de_semana", "ponto_facultativo"]:
                is_letivo = False

        chave_mes_dia = data_iso[5:]
        nome_feriado = FERIADOS_FIXOS_BR.get(chave_mes_dia)
        if not is_letivo:
            if nome_feriado:
                evento_rotulo = f"Feriado Nacional ({nome_feriado})"
                prato = f"Feriado ({nome_feriado}) — Refeitório Fechado"
            elif evento_nome.lower() == "feriado":
                evento_rotulo = "Feriado Nacional"
                prato = "Feriado — Refeitório Fechado"
            elif "recesso" in evento_nome.lower():
                evento_rotulo = "Recesso Escolar"
                prato = "Recesso Escolar — Refeitório Fechado"
            else:
                evento_rotulo = evento_nome.replace("_", " ").title()
                prato = f"Não Letivo ({evento_rotulo}) — Refeitório Fechado"
            status_calendario = f"Não Letivo / {evento_rotulo}"
        else:
            evento_rotulo = "Letivo Regular"
            status_calendario = "Dia Letivo Regular"

        # Clima previsto para o dia e rastreabilidade da fonte meteorológica
        clima_row = df_clima_futuro[df_clima_futuro["data_iso"] == data_iso] if not df_clima_futuro.empty else pd.DataFrame()
        if not clima_row.empty:
            precip = float(clima_row.iloc[0]["precipitation_sum"])
            temp_max = float(clima_row.iloc[0]["temperature_2m_max"])
            fonte_clima_dia = str(clima_row.iloc[0].get("fonte_clima", "Satélite Open-Meteo"))
        else:
            precip = 0.0
            temp_max = mediana_temp
            fonte_clima_dia = "Histórico Mediano (Offline)"

        # Planejamento da nutricionista e rastreabilidade do cardápio
        plan_row = df_planejamento[df_planejamento["data_iso"] == data_iso]
        if not is_letivo:
            prot = "—"
            preparo = "—"
            status_cardapio = "Não Aplicável (Dia Não Letivo)"
        elif not plan_row.empty and usar_cardapio and pd.notna(plan_row.iloc[0].get("prato_principal")) and str(plan_row.iloc[0].get("prato_principal")).strip() not in ["", "nan", "None"]:
            prato = str(plan_row.iloc[0].get("prato_principal"))
            prot = str(plan_row.iloc[0].get("proteina_tipo", "frango")).lower()
            preparo = str(plan_row.iloc[0].get("preparo_tipo", "assado"))
            status_cardapio = "Oficial (Cadastrado)"
        elif not row_fato.empty and usar_cardapio and pd.notna(row_fato.iloc[0].get("prato_principal")) and str(row_fato.iloc[0].get("prato_principal")).strip() not in ["", "nan", "None"]:
            prato = str(row_fato.iloc[0].get("prato_principal"))
            prot = str(row_fato.iloc[0].get("proteina_tipo", "frango")).lower()
            preparo = str(row_fato.iloc[0].get("preparo_tipo", "assado"))
            status_cardapio = "Oficial (Histórico)"
        else:
            prato = "Padrão Escolar Regular"
            prot = "frango"
            preparo = "assado"
            status_cardapio = "Padrão Neutro (Não Cadastrado)" if usar_cardapio else "Desativado no Cenário"
            
        # Quantitativo Institucional / Nutricionista e rastreabilidade da meta humana
        val_nutri = None
        origem_nutri = "Não informado"
        val_nutri_informado = None
        status_nutri = "Não Informado (Operação Autônoma)"
        if not is_letivo:
            status_nutri = "Não Aplicável (Dia Não Letivo)"
        elif usar_nutri and is_letivo:
            if not plan_row.empty and pd.notna(plan_row.iloc[0].get("planejado_nutri")) and float(plan_row.iloc[0].get("planejado_nutri", 0)) > 0:
                val_nutri = float(plan_row.iloc[0].get("planejado_nutri"))
                val_nutri_informado = val_nutri
                origem_nutri = "Planejamento Nutrição (Produção)"
                status_nutri = "Oficial (Planejado pela Gestão)"
            elif not row_fato.empty and pd.notna(row_fato.iloc[0].get("planejado_nutri")) and float(row_fato.iloc[0].get("planejado_nutri", 0)) > 0:
                val_nutri = float(row_fato.iloc[0].get("planejado_nutri"))
                val_nutri_informado = val_nutri
                origem_nutri = "Quantitativo Institucional (Reservas)"
                status_nutri = "Quantitativo Institucional"
            elif not row_fato.empty and pd.notna(row_fato.iloc[0].get("reservas")) and float(row_fato.iloc[0].get("reservas", 0)) > 0:
                val_nutri = float(row_fato.iloc[0].get("reservas"))
                val_nutri_informado = val_nutri
                origem_nutri = "Quantitativo Institucional (Reservas)"
                status_nutri = "Quantitativo Institucional"
        elif hz == "t2":
            status_nutri = "Não Usado nas Compras (T+2)"
        else:
            status_nutri = "Desativado no Cenário"

        lag_5 = lag_10 = media_5d = 0.0
        pred_ia_base = pred_base = margem_valor = 0.0
        linha = atributos[hz][atributos[hz]["data"] == pd.Timestamp(dt)] if is_letivo and not atributos[hz].empty else pd.DataFrame()

        if not linha.empty:
            cenario_ia = cenario_oficial(hz) if cfg["numero"] == cenario_oficial("t1") else cenario_da_ia(cfg, hz, eh_hibrido)
            linha = aplicar_entradas(linha, cenario_ia, val_nutri, precip, temp_max)
            pred_ia_base = prever(modelo_congelado(hz, cenario_ia), linha, hz, cenario_ia)
            lag_5, lag_10 = (float(linha[c].iloc[0]) if pd.notna(linha[c].iloc[0]) else 0.0 for c in ("mesmo_dia_ha_1_ocorrencias", "mesmo_dia_ha_2_ocorrencias"))
            media_5d = float(linha["media_5d"].iloc[0])

            if eh_manual and val_nutri is not None:
                pred_base, politica = val_nutri, "manual"
            elif eh_hibrido and val_nutri is not None:
                pred_base, politica = float(combinar_hibrido([pred_ia_base], [val_nutri])[0]), "hibrido"
            else:
                pred_base, politica = pred_ia_base, ("t2" if hz == "t2" else "ia")
            margem_valor = calcular_margem(tipo_margem, politica, margem_adicional)

        # Regra de Resiliência Climática: se tempestade extrema (>= 15 mm), desativa margem aditiva
        if usar_clima and precip >= 15.0:
            margem_valor = 0.0 # Modo de Contenção de Desperdício

        pred_final = max(0, pred_base + margem_valor)
        
        # Alertas operacionais inteligentes (sem emojis)
        alertas = []
        if is_letivo and linha.empty:
            alertas.append("Data fora do escopo do modelo (sábado letivo ou histórico insuficiente)")
        if is_letivo and (eh_hibrido or eh_manual) and val_nutri is None:
            alertas.append(f"Quantitativo institucional não informado: usando apenas o {NOME_PREPARO}")
        if not is_letivo:
            pred_base = 0.0
            margem_valor = 0.0
            pred_final = 0.0
            pred_ia_base = 0.0
            alertas.append(f"{evento_rotulo} — Refeitório Fechado (0 refeições)")
        else:
            if dia_sem_num == 4:
                alertas.append("Sexta-feira: retração sistemática de comparecimento (~20% menor)")
            if usar_clima and precip >= 15.0:
                alertas.append(f"Alerta tempestade ({precip:.1f} mm): modo contenção ativado (margem zerada para evitar desperdício)")
            elif precip >= 15.0:
                alertas.append(f"Chuva forte prevista ({precip:.1f} mm): avaliar o risco de queda de comparecimento")
            elif precip >= 8.0:
                alertas.append(f"Chuva prevista ({precip:.1f} mm): risco moderado de evasão")
            if usar_cardapio and cenario_ia >= 2:
                if status_cardapio == "Padrão Neutro (Não Cadastrado)":
                    alertas.append("Cardápio não cadastrado: previsão sem o efeito do prato")
                elif "bovina" in prot or "feijoada" in prato.lower():
                    alertas.append("Cardápio com proteína bovina: alta atratividade histórica")
                elif "peixe" in prot or "ovo" in prot:
                    alertas.append("Proteína com menor atratividade histórica")
            
        linhas_pred.append({
            "data_dt": dt,
            "data": data_formatada,
            "data_iso": data_iso,
            "dia_semana": dia_sem_nome,
            "is_dia_letivo": is_letivo,
            "evento_tipo": evento_rotulo,
            "status_calendario": status_calendario,
            "horizonte": "Semana t+1 (Preparo)" if horizonte == "Semana_t1" else "Semana t+2 (Compras)",
            "prato_principal": prato if (usar_cardapio or not is_letivo) else "Cardápio Neutro",
            "proteina_tipo": prot.capitalize() if is_letivo else "—",
            "preparo_tipo": preparo.capitalize() if is_letivo else "—",
            "status_cardapio": status_cardapio,
            "chuva_mm": round(precip, 1),
            "temp_max_c": round(temp_max, 1),
            "status_clima": fonte_clima_dia,
            "estimativa_nutri": round(val_nutri_informado) if (val_nutri_informado is not None and is_letivo) else None,
            "status_nutri": status_nutri,
            "origem_nutri": origem_nutri,
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
            "Média Móvel 5d": round(media_5d, 1),
            "Chuva (mm)": round(precip, 1),
            "Temp Máx (°C)": round(temp_max, 1),
            "Proteína": prot.capitalize(),
            "Nutri Planejado": round(val_nutri_informado, 1) if val_nutri_informado is not None else "Não informado"
        })
        
    df_resultado = pd.DataFrame(linhas_pred)
    df_features = pd.DataFrame(linhas_features)
    salvar_predicoes_geradas(df_resultado)
    return df_resultado, df_features

def carregar_gold_passado(horizonte, origem):
    """Linhas da Gold emitidas até a sexta de corte, usadas para buscar efeitos já conhecidos."""
    gold = carregar_gold(horizonte)
    return gold[gold["origem"] <= origem].sort_values("origem")


def obter_clima_emissao(data_alvo, horizonte):
    """Devolve a previsão de clima disponível na emissão. Nunca usa clima observado."""
    arquivada = linha_arquivada(data_alvo, horizonte)
    if not arquivada.empty and pd.notna(arquivada["previsao_chuva_mm"].iloc[0]):
        return float(arquivada["previsao_chuva_mm"].iloc[0]), float(arquivada["previsao_temp_max"].iloc[0]), "Previsão Arquivada na Emissão"
    if pd.Timestamp(data_alvo).normalize() >= pd.Timestamp.now().normalize():
        return obter_clima_data(pd.Timestamp(data_alvo).strftime("%Y-%m-%d"))
    return None, None, "Previsão não arquivada para esta data"


def simular_previsao_data(
    data_alvo,
    objetivo="consumo",
    cenario_escolhido=CENARIO_PADRAO,
    modelo_escolhido=MODELO_HIBRIDO,
    tipo_margem=MARGEM_PADRAO,
    margem_customizada=0.0,
    proteina_custom=None,
    prato_custom=None,
    preparo_custom=None,
    chuva_custom=None,
    temp_custom=None,
    nutri_custom=None
):
    """
    Simula uma data como se a decisão fosse tomada na sexta de corte daquele horizonte.
    Datas anteriores a 01/09/2026 usam walk-forward treinado só até a sexta de corte. As demais usam o
    modelo congelado em 31/08/2026. Em ambos os casos, o clima é sempre o previsto, nunca o observado.
    """
    dt_alvo = pd.to_datetime(data_alvo)
    data_iso = dt_alvo.strftime("%Y-%m-%d")
    data_formatada = dt_alvo.strftime("%d/%m/%Y")
    dia_sem = dt_alvo.dayofweek
    dia_sem_nome = DIAS_SEMANA_PT.get(dia_sem, "Dia Útil")
    is_fim_semana = (dia_sem >= 5)
    hz = "t1" if objetivo == "consumo" else "t2"

    df_fato = carregar_fato_diaria_completa()
    df_planejamento = carregar_planejamento_nutricionista()
    origem = origem_da_data(dt_alvo, hz)

    # 1. Histórico disponível até a sexta de corte
    df_hist = df_fato[(df_fato["data_dt"] < origem) & (df_fato["consumo"].notna()) & (df_fato["consumo"] > 0)]

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
    prot_utilizada = str(proteina_custom if proteina_custom is not None else (prot_base if tem_cardapio_base else "sem cardápio")).lower()
    prep_utilizado = preparo_custom if preparo_custom is not None else (prep_base if tem_cardapio_base else "sem cardápio")

    # Clima previsto na emissão
    chuva_base, temp_base, fonte_clima = obter_clima_emissao(dt_alvo, hz)
    tem_clima_base = chuva_base is not None and temp_base is not None
    chuva_utilizada = float(chuva_custom if chuva_custom is not None else (chuva_base if tem_clima_base else 0.0))
    temp_utilizada = float(temp_custom if temp_custom is not None else (temp_base if tem_clima_base else 24.5))

    # Quantitativo institucional: existe apenas no T+1
    tem_nutri_base = False
    nutri_base = None
    if hz == "t1":
        if not row_plan.empty and pd.notna(row_plan.iloc[0].get("planejado_nutri")) and float(row_plan.iloc[0].get("planejado_nutri", 0)) > 0:
            tem_nutri_base = True
            nutri_base = float(row_plan.iloc[0]["planejado_nutri"])
        elif not row_fato.empty and pd.notna(row_fato.iloc[0].get("planejado_nutri")) and float(row_fato.iloc[0].get("planejado_nutri", 0)) > 0:
            tem_nutri_base = True
            nutri_base = float(row_fato.iloc[0]["planejado_nutri"])
        elif not row_fato.empty and pd.notna(row_fato.iloc[0].get("reservas")) and float(row_fato.iloc[0].get("reservas", 0)) > 0:
            tem_nutri_base = True
            nutri_base = float(row_fato.iloc[0]["reservas"])
    nutri_utilizado = float(nutri_custom) if (nutri_custom is not None and hz == "t1") else nutri_base

    # Consumo Real se conhecido no histórico
    consumo_real = float(row_fato.iloc[0]["consumo"]) if not row_fato.empty and pd.notna(row_fato.iloc[0].get("consumo")) and float(row_fato.iloc[0]["consumo"]) > 0 else None

    # 3. Configuração do Cenário
    cen_cfg = numero_do_cenario(cenario_escolhido)
    usar_clima = cen_cfg["usar_clima"]
    usar_cardapio = cen_cfg["usar_cardapio"]
    eh_manual = "Manual" in modelo_escolhido and hz == "t1"
    eh_hibrido = "Híbrido" in modelo_escolhido and hz == "t1"
    usar_nutri = hz == "t1" and (cen_cfg["usar_nutri"] or eh_hibrido or eh_manual)
    cenario_ia = cenario_da_ia(cen_cfg, hz, eh_hibrido)
    algoritmo = MODELOS_SIMULADOR.get(modelo_escolhido, {}).get("algoritmo")

    # 4. Atributos na sexta de corte
    linha = linha_arquivada(dt_alvo, hz)
    if linha.empty and origem >= pd.Timestamp("2026-10-30"):
        emissao = atributos_da_emissao(df_fato, origem, hz)
        linha = emissao[emissao["data"] == dt_alvo] if not emissao.empty else emissao

    pred_ia = pred_base = margem_val = 0.0
    media_5d = lag5 = lag10 = 0.0
    regime_modelo = "Sem previsão para esta data"
    if not linha.empty:
        linha = aplicar_entradas(linha, cenario_ia, nutri_utilizado, chuva_utilizada if tem_clima_base or chuva_custom is not None else None, temp_utilizada)
        if proteina_custom is not None and usar_cardapio:
            gold = carregar_gold_passado(hz, origem)
            efeitos = gold.loc[gold["proteina"] == str(proteina_custom).lower(), "efeito_proteina"].dropna()
            linha = linha.assign(efeito_proteina=float(efeitos.iloc[-1]) if not efeitos.empty else np.nan, sem_cardapio=False)
        if dt_alvo < pd.Timestamp(FIM_TREINO_CONGELADO):
            modelo = modelo_walk_forward(hz, cenario_ia, origem, algoritmo)
            regime_modelo = f"Walk-forward até {origem.strftime('%d/%m/%Y')}"
        else:
            modelo = modelo_congelado(hz, cenario_ia, algoritmo)
            regime_modelo = "Modelo congelado em 31/08/2026"
        if modelo is not None:
            pred_ia = prever(modelo, linha, hz, cenario_ia)
            media_5d = float(linha["media_5d"].iloc[0])
            lag5, lag10 = (float(linha[c].iloc[0]) if pd.notna(linha[c].iloc[0]) else 0.0 for c in ("mesmo_dia_ha_1_ocorrencias", "mesmo_dia_ha_2_ocorrencias"))
            if eh_manual and nutri_utilizado is not None:
                pred_base, politica = nutri_utilizado, "manual"
            elif eh_hibrido and nutri_utilizado is not None:
                pred_base, politica = float(combinar_hibrido([pred_ia], [nutri_utilizado])[0]), "hibrido"
            else:
                pred_base, politica = pred_ia, ("t2" if hz == "t2" else "ia")
            margem_val = margem_do_modelo(tipo_margem, politica, margem_customizada, hz, cenario_ia, algoritmo)

    if usar_clima and chuva_utilizada >= 15.0:
        margem_val = 0.0

    pred_final = max(0, pred_base + margem_val)

    is_letivo_sim = True
    evento_tipo_sim = "Letivo Regular"
    if not row_fato.empty:
        is_letivo_sim = bool(row_fato.iloc[0].get("is_dia_letivo", True))
        evento_tipo_sim = str(row_fato.iloc[0].get("evento_tipo", "Letivo Regular"))
        status_leg = str(row_fato.iloc[0].get("status_legal", "")).lower().strip()
        if status_leg == "nao_letivo" or evento_tipo_sim.lower() in ["feriado", "recesso", "ferias", "sem_atividade_letiva", "final_de_semana", "ponto_facultativo"]:
            is_letivo_sim = False
    elif is_fim_semana:
        is_letivo_sim = False
        evento_tipo_sim = "final_de_semana"

    chave_mes_dia = data_iso[5:]
    nome_feriado = FERIADOS_FIXOS_BR.get(chave_mes_dia)
    if not is_letivo_sim:
        if nome_feriado:
            evento_tipo_sim = f"Feriado Nacional ({nome_feriado})"
            prato_utilizado = f"Feriado ({nome_feriado}) — Refeitório Fechado"
        elif evento_tipo_sim.lower() == "feriado":
            evento_tipo_sim = "Feriado Nacional"
            prato_utilizado = "Feriado — Refeitório Fechado"
        elif "recesso" in evento_tipo_sim.lower():
            evento_tipo_sim = "Recesso Escolar"
            prato_utilizado = "Recesso Escolar — Refeitório Fechado"
        elif is_fim_semana:
            evento_tipo_sim = "Fim de Semana"
            prato_utilizado = "Fim de Semana — Refeitório Fechado"
        else:
            evento_tipo_sim = evento_tipo_sim.replace("_", " ").title()
            prato_utilizado = f"Não Letivo ({evento_tipo_sim}) — Refeitório Fechado"
        pred_base = 0.0
        margem_val = 0.0
        pred_final = 0.0
        pred_ia = 0.0

    alertas = []
    if not is_letivo_sim:
        alertas.append(f"{evento_tipo_sim}: Restaurante sem atendimento regular (0 refeições sugeridas).")
    elif is_fim_semana:
        alertas.append("Fim de semana (sábado/domingo): refeitório fechado ou com cardápio excepcional.")
    elif linha.empty:
        alertas.append("Data fora do escopo do modelo (sábado letivo ou histórico insuficiente).")
    elif dia_sem == 4:
        alertas.append("Sexta-feira: retração histórica de ~20% no comparecimento à tarde.")
    elif dia_sem == 2:
        alertas.append("Quarta-feira: dia de maior comparecimento discente (pico semanal com aulas técnicas).")

    if usar_clima and chuva_utilizada >= 15.0:
        alertas.append(f"Alerta Tempestade ({chuva_utilizada:.1f} mm): Margem aditiva zerada para evitar desperdício de insumos.")
    elif usar_clima and chuva_utilizada >= 5.0:
        alertas.append(f"Chuva prevista ({chuva_utilizada:.1f} mm): Queda esperada na adesão de alunos.")

    if (eh_hibrido or eh_manual) and nutri_utilizado is None and is_letivo_sim:
        alertas.append(f"Quantitativo institucional não informado: resultado calculado apenas com o {NOME_PREPARO}.")

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
        "regime_modelo": regime_modelo,
        "origem": origem,
        "pred_ia": round(pred_ia, 1),
        "pred_base": round(pred_base, 1),
        "margem_val": round(margem_val, 1),
        "pred_final": int(round(pred_final)),
        "mm5": round(media_5d, 1),
        "lag5": round(lag5, 1),
        "lag10": round(lag10, 1),
        "alertas": alertas,
        "vetor_features": linha.iloc[0].to_dict() if not linha.empty else {}
    }
