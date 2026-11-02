"""Bateria de Testes Adversariais de Não-Vazamento Temporal nos Modelos Preditivos.

Exercita a função real de avaliação (avaliar_um_algoritmo) em 4 dimensões:
  1. Isolamento de Variáveis: alvos, oráculos e colunas de controle derivadas do resultado
     não podem estar na matriz de preditores X; T+2 não pode ter quantitativo institucional.
  2. Fronteira Temporal: registra cada chamada real de fit e confirma max(data_treino) < origem.
  3. Seleção do Teste: todo dia de 2026 com consumo precisa ser avaliado. Nenhuma coluna
     derivada do resultado pode retirar dias do teste.
  4. Invariância Adversarial: corrompe consumo, alvo e rótulos a partir da sexta de corte
     e mede a diferença real entre as previsões.

Alcance: os testes cobrem consumo, rótulos e oráculos. Não comprovam o horário em que reservas,
plano da nutricionista ou cardápio foram registrados, que dependem da declaração do autor.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional

from .config import GOLD, RESULTADOS, SEMENTE_ALEATORIA
from .models import (
    avaliar_um_algoritmo,
    obter_catalogo_modelos,
    definir_cenarios_features
)

PASTA_AUDITORIA = RESULTADOS / "auditoria"
PASTA_AUDITORIA.mkdir(parents=True, exist_ok=True)

CENARIO_AUDITADO = "Cenario_2_Com_Cardapio"
COLUNAS_PROIBIDAS = {
    "consumo", "alvo_relativo", "tratamento", "usar_no_treino", "tipo_sugerido",
    "oraculo_chuva_mm", "oraculo_chuva_horas", "oraculo_temp_max", "oraculo_vento_max"
}
MODELOS_INVARIANCIA = ["Ridge", "Ensemble_Bagging_ExtraTrees", "Ensemble_Boosting_Gradient"]
ORIGENS_AMOSTRA = ["2024-03-08", "2024-09-13", "2025-05-09", "2026-03-06", "2026-09-04"]


class ModeloEspiao:
    """Registra as linhas recebidas no fit e no predict da avaliação real."""
    def __init__(self, registros: list):
        self.registros = registros

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.registros.append({"treino": X.index.tolist(), "alvo": []})
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.registros[-1]["alvo"] = X.index.tolist()
        return np.ones(len(X))


def registrar_chamadas(gold: pd.DataFrame, horizonte: str, colunas: List[str], origens: list, avaliador: Callable) -> list:
    """Executa o avaliador com o modelo espião e devolve as linhas de cada emissão."""
    registros = []
    info = {"construtor": lambda: ModeloEspiao(registros), "familia": "Auditoria", "tipo_ensemble": "Auditoria"}
    avaliador("Espiao", info, horizonte, "Auditoria", colunas, gold, origens)
    return registros


def origens_de_2026(gold: pd.DataFrame) -> list:
    return sorted(gold.loc[gold["ano"] == 2026, "origem"].unique())


def testar_isolamento_variaveis_alvo(cenarios: Dict[str, List[str]], gold_t1: pd.DataFrame, gold_t2: pd.DataFrame) -> dict:
    """Verifica se alvos, oráculos e colunas de controle não constam nas matrizes de preditores."""
    erros = []
    for nome_cenario, cols in cenarios.items():
        vazamentos = set(cols).intersection(COLUNAS_PROIBIDAS)
        if vazamentos:
            erros.append(f"{nome_cenario} contém colunas proibidas: {sorted(vazamentos)}")

    # Garantir que em T+2 não há quantitativo institucional
    cols_t2 = gold_t2.columns
    if "plano_nutricionista" in cols_t2 or "reserva_discente" in cols_t2 or "rel_plano_humano" in cols_t2:
        erros.append("Gold T2 contém atributos de planejamento humano!")

    sucesso = len(erros) == 0
    return {
        "teste": "1. Isolamento de Variáveis Alvo e Restrições de Horizonte",
        "sucesso": sucesso,
        "erros": erros,
        "detalhe": "Nenhum alvo, oráculo ou coluna de controle nas matrizes X. T2 sem quantitativo institucional." if sucesso else "; ".join(erros)
    }


def testar_fronteira_temporal_todas_semanas(gold_t1: pd.DataFrame, gold_t2: pd.DataFrame, avaliador: Callable = avaliar_um_algoritmo) -> dict:
    """Registra os fits reais do walk-forward e verifica max(data_treino) < origem em cada emissão."""
    colunas = definir_cenarios_features()[CENARIO_AUDITADO]
    erros = []
    emissoes = 0
    sem_dias = []

    for nome_horiz, df in (("t1", gold_t1), ("t2", gold_t2)):
        cols = [c for c in colunas if c in df.columns]
        origens = origens_de_2026(df)
        registros = registrar_chamadas(df, nome_horiz, cols, origens, avaliador)
        vistas = set()
        for reg in registros:
            origem = df.loc[reg["alvo"], "origem"].iloc[0]
            vistas.add(origem)
            emissoes += 1
            max_treino = df.loc[reg["treino"], "data"].max()
            if max_treino >= origem:
                erros.append(f"{nome_horiz} na emissão {origem.date()} treinou com a data {max_treino.date()}")
        sem_dias += [f"{nome_horiz} {pd.Timestamp(o).date()}" for o in origens if o not in vistas]

    sucesso = len(erros) == 0
    return {
        "teste": "2. Fronteira Temporal nas Chamadas Reais de Treino (max(treino) < origem)",
        "sucesso": sucesso,
        "semanas_auditadas": emissoes,
        "emissoes_sem_dias_avaliaveis": sem_dias,
        "erros": erros,
        "detalhe": f"{emissoes} emissões com fit registrado; {len(sem_dias)} sem dias avaliáveis: {sem_dias}." if sucesso else "; ".join(erros[:5])
    }


def testar_selecao_do_teste(gold_t1: pd.DataFrame, avaliador: Callable = avaliar_um_algoritmo) -> dict:
    """Confirma que todos os dias de 2026 com consumo entram no teste."""
    cols = [c for c in definir_cenarios_features()[CENARIO_AUDITADO] if c in gold_t1.columns]
    origens = origens_de_2026(gold_t1)
    registros = registrar_chamadas(gold_t1, "t1", cols, origens, avaliador)

    avaliados = {i for reg in registros for i in reg["alvo"]}
    esperados = set(gold_t1[gold_t1["origem"].isin(origens) & gold_t1["consumo"].notna()].index)
    ausentes = gold_t1.loc[sorted(esperados - avaliados), "data"].dt.strftime("%Y-%m-%d").tolist()

    sucesso = len(ausentes) == 0
    erros = [f"Dias com consumo fora do teste: {ausentes}"] if ausentes else []
    return {
        "teste": "3. Seleção do Teste sem Colunas Derivadas do Resultado",
        "sucesso": sucesso,
        "dias_avaliados": len(avaliados),
        "erros": erros,
        "detalhe": f"{len(avaliados)} de {len(esperados)} dias com consumo foram avaliados." if sucesso else "; ".join(erros)
    }


def corromper_futuro(gold: pd.DataFrame, origem: pd.Timestamp, gerador: np.random.Generator) -> pd.DataFrame:
    """Substitui por ruído tudo o que só seria conhecido depois do corte."""
    corrompido = gold.copy()
    eh_futuro = corrompido["data"] >= origem
    tem_consumo = eh_futuro & corrompido["consumo"].notna()
    corrompido.loc[tem_consumo, "consumo"] = gerador.integers(1, 500, tem_consumo.sum()).astype(float)
    corrompido.loc[tem_consumo, "alvo_relativo"] = gerador.uniform(0.2, 2.5, tem_consumo.sum())
    corrompido.loc[eh_futuro, "usar_no_treino"] = ~corrompido.loc[eh_futuro, "usar_no_treino"]
    corrompido.loc[eh_futuro, "tratamento"] = "corrompido"
    return corrompido


def testar_invariancia_adversarial_modelos(
    gold_t1: pd.DataFrame,
    origens_amostra: List[str],
    colunas: Optional[List[str]] = None,
    avaliador: Callable = avaliar_um_algoritmo,
    preparar: Callable = lambda gold: gold
) -> dict:
    """Corrompe o futuro e mede a diferença real entre as previsões da avaliação."""
    gerador = np.random.default_rng(SEMENTE_ALEATORIA)
    catalogo = obter_catalogo_modelos()
    if colunas is None:
        colunas = [c for c in definir_cenarios_features()[CENARIO_AUDITADO] if c in gold_t1.columns]

    erros = []
    detalhes = []

    for origem_str in origens_amostra:
        origem = pd.Timestamp(origem_str)
        corrompido = corromper_futuro(gold_t1, origem, gerador)
        maior_dif = 0.0

        for nome_m in MODELOS_INVARIANCIA:
            info = catalogo[nome_m]
            _, pred_orig = avaliador(nome_m, info, "t1", CENARIO_AUDITADO, colunas, preparar(gold_t1), [origem])
            _, pred_corr = avaliador(nome_m, info, "t1", CENARIO_AUDITADO, colunas, preparar(corrompido), [origem])
            if not pred_orig:
                erros.append(f"Emissão {origem_str} sem dias avaliáveis para {nome_m}")
                continue
            dif = np.abs(np.array([p["previsao"] for p in pred_orig]) - np.array([p["previsao"] for p in pred_corr]))
            maior_dif = max(maior_dif, float(dif.max()))
            if dif.max() > 1e-6:
                erros.append(f"Vazamento detectado em {nome_m} na emissão {origem_str}: diferença máxima = {dif.max():.4f}")

        detalhes.append(f"Emissão {origem.strftime('%d/%m/%Y')}: maior diferença entre previsões = {maior_dif:.6f} refeições.")

    sucesso = len(erros) == 0
    return {
        "teste": "4. Invariância Adversarial com Consumo, Alvo e Rótulos Futuros Corrompidos",
        "sucesso": sucesso,
        "erros": erros,
        "detalhes": detalhes,
        "conclusao": f"{len(origens_amostra)} emissões testadas em {len(MODELOS_INVARIANCIA)} modelos." if sucesso else "FALHA DE ISOLAMENTO"
    }


def executar_auditoria_completa_vazamento() -> dict:
    """Executa a bateria completa de testes de vazamento e salva o relatório."""
    gold_t1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    gold_t2 = pd.read_parquet(GOLD / "gold_t2.parquet")
    cenarios = definir_cenarios_features()

    testes = [
        testar_isolamento_variaveis_alvo(cenarios, gold_t1, gold_t2),
        testar_fronteira_temporal_todas_semanas(gold_t1, gold_t2),
        testar_selecao_do_teste(gold_t1),
        testar_invariancia_adversarial_modelos(gold_t1, ORIGENS_AMOSTRA)
    ]

    relatorio = {
        "data_auditoria": pd.Timestamp.now().strftime("%d/%m/%Y %H:%M:%S"),
        "status_geral": "APROVADO NOS TESTES EXECUTADOS" if all(t["sucesso"] for t in testes) else "REPROVADO",
        "alcance": "Cobre consumo, rótulos e oráculos na avaliação da pesquisa. Não cobre o aplicativo.",
        "testes": testes
    }

    with open(PASTA_AUDITORIA / "relatorio_teste_vazamento_modelos.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)

    return relatorio


if __name__ == "__main__":
    rel = executar_auditoria_completa_vazamento()
    print("\n=======================================================")
    print(f"RELATÓRIO DE AUDITORIA DE VAZAMENTO: {rel['status_geral']}")
    print("=======================================================")
    for teste in rel["testes"]:
        tag = "[OK] APROVADO" if teste["sucesso"] else "[FALHA] REPROVADO"
        print(f"\n{tag} - {teste['teste']}")
        if "detalhe" in teste:
            print(f"  -> {teste['detalhe']}")
        if "detalhes" in teste:
            for d in teste["detalhes"]:
                print(f"  * {d}")
        if teste["erros"]:
            for e in teste["erros"]:
                print(f"  ! ERRO: {e}")
