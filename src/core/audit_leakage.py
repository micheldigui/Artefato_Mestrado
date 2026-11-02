"""Módulo de Auditoria da Camada Gold: confere se os atributos respeitam a sexta de corte.

Cobre atributos derivados de consumo, rótulos e clima observado. A avaliação dos modelos é
auditada em test_zero_leakage_models.py.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .config import GOLD, RESULTADOS, SEMENTE_ALEATORIA, SILVER
from .gold_features import construir_gold

# Colunas que dependem do resultado do dia-alvo por construção: não são preditores e não podem selecionar teste.
COLUNAS_RESULTADO_E_ROTULO = {
    "consumo",
    "alvo_relativo",
    "tratamento",
    "usar_no_treino",
    "tipo_sugerido",
    "oraculo_chuva_mm",
    "oraculo_chuva_horas",
    "oraculo_temp_max",
    "oraculo_vento_max"
}


def testar_estrutura_operacional(t1: pd.DataFrame, t2: pd.DataFrame) -> dict:
    """Valida se as origens temporais, antecedências e restrições de cada horizonte estão corretas."""
    erros = []
    
    # 1. Origem em sextas-feiras
    if not t1["origem"].dt.dayofweek.eq(4).all():
        erros.append("t1 possui emissões fora de sextas-feiras")
    if not t2["origem"].dt.dayofweek.eq(4).all():
        erros.append("t2 possui emissões fora de sextas-feiras")
        
    # 2. Faixas de antecedência
    if not t1["antecedencia_dias"].between(3, 7).all():
        erros.append(f"t1 possui antecedências fora de [3, 7] dias: min={t1['antecedencia_dias'].min()}, max={t1['antecedencia_dias'].max()}")
    if not t2["antecedencia_dias"].between(10, 14).all():
        erros.append(f"t2 possui antecedências fora de [10, 14] dias: min={t2['antecedencia_dias'].min()}, max={t2['antecedencia_dias'].max()}")
        
    # 3. Consumo estritamente anterior à sexta
    if not t1["ultimo_consumo_usado"].lt(t1["origem"]).all():
        erros.append("t1 utilizou consumo no mesmo dia ou após a sexta de corte!")
    if not t2["ultimo_consumo_usado"].lt(t2["origem"]).all():
        erros.append("t2 utilizou consumo no mesmo dia ou após a sexta de corte!")
        
    # 4. Proibição de dados humanos de curto prazo no horizonte t2 (Compras)
    if "plano_nutricionista" in t2.columns:
        erros.append("Vazamento: plano_nutricionista presente em t2 (Compras)!")
    if "reserva_discente" in t2.columns:
        erros.append("Vazamento: reserva_discente presente em t2 (Compras)!")
        
    # 5. Unicidade de chaves
    if t1.duplicated(["origem", "data"]).any():
        erros.append("Chave (origem, data) duplicada em t1")
    if t2.duplicated(["origem", "data"]).any():
        erros.append("Chave (origem, data) duplicada em t2")
        
    sucesso = len(erros) == 0
    return {
        "teste": "Estrutura Operacional e Restrições de Horizonte",
        "sucesso": sucesso,
        "erros": erros,
        "detalhe": "Origens em sextas, antecedências corretas e horizonte t2 estritamente autônomo." if sucesso else "; ".join(erros)
    }


def testar_qualidade_dados(t1: pd.DataFrame, t2: pd.DataFrame) -> dict:
    """Valida ausência de valores infinitos, estabilidade de nível e ausência de atributos constantes."""
    erros = []
    
    for nome, df in [("t1", t1), ("t2", t2)]:
        numericas = df.select_dtypes("number")
        if np.isinf(numericas.to_numpy(dtype=float, na_value=np.nan)).any():
            erros.append(f"{nome} possui valores infinitos")
        if not df["nivel_base"].gt(0).all():
            erros.append(f"{nome} possui nível basal <= 0")
        if df["consumo"].dropna().lt(0).any():
            erros.append(f"{nome} possui valores negativos de consumo")
            
        # Atributos constantes no subconjunto de treino
        treino = df[df["usar_no_treino"]]
        constantes = [c for c in treino.columns if treino[c].nunique(dropna=True) <= 1 and c not in ("horizonte", "usar_no_treino")]
        if constantes:
            erros.append(f"{nome} possui colunas constantes no treino: {constantes}")
            
    sucesso = len(erros) == 0
    return {
        "teste": "Qualidade Numérica e Ausência de Constantes",
        "sucesso": sucesso,
        "erros": erros,
        "detalhe": "Sem valores infinitos, níveis basais positivos e sem colunas nulas/constantes." if sucesso else "; ".join(erros)
    }


def testar_vazamento_temporal_adversarial(fato: pd.DataFrame, t1: pd.DataFrame, t2: pd.DataFrame, origens_teste: list) -> dict:
    """Corrompe consumo, registros e clima observado do futuro e valida se os preditores da emissão mudam."""
    gerador = np.random.default_rng(SEMENTE_ALEATORIA)
    colunas_preditoras = lambda df: [c for c in df.columns if c not in COLUNAS_RESULTADO_E_ROTULO]
    
    erros = []
    detalhes_origens = []
    
    for origem in origens_teste:
        data_origem = pd.to_datetime(origem)
        fato_embaralhado = fato.copy()
        
        # Seleciona tudo o que vem da sexta em diante e substitui por ruído aleatório
        eh_futuro = fato_embaralhado["data"] >= data_origem
        tem_consumo_futuro = eh_futuro & fato_embaralhado["consumo"].notna()
        n_alterados = tem_consumo_futuro.sum()
        
        if n_alterados > 0:
            fato_embaralhado.loc[tem_consumo_futuro, "consumo"] = gerador.integers(1, 500, n_alterados).astype(float)
            fato_embaralhado.loc[tem_consumo_futuro, "status_consumo"] = "atipico"
            fato_embaralhado.loc[eh_futuro, "chuva_mm"] = gerador.uniform(0, 100, eh_futuro.sum())
            
        # Reconstrói a Camada Gold com o futuro contaminado por ruído
        gold_recalculado = construir_gold(fato_embaralhado)
        
        # Valida se os preditores calculados para essa sexta-feira permaneceram idênticos
        erros_antes = len(erros)
        for nome_horiz, df_original in (("t1", t1), ("t2", t2)):
            original_origem = df_original[df_original["origem"] == data_origem].reset_index(drop=True)
            recalculado_origem = gold_recalculado[nome_horiz][gold_recalculado[nome_horiz]["origem"] == data_origem].reset_index(drop=True)
            
            if original_origem.empty:
                continue
                
            pred_orig = colunas_preditoras(original_origem)
            pred_recalc = colunas_preditoras(recalculado_origem)
            
            try:
                pd.testing.assert_frame_equal(
                    original_origem[pred_orig],
                    recalculado_origem[pred_recalc],
                    check_dtype=False
                )
            except AssertionError as exc:
                erros.append(f"Vazamento detectado em {nome_horiz} na data {origem}: {str(exc)[:200]}")
                
        situacao = "preditores idênticos" if len(erros) == erros_antes else "preditores alterados"
        detalhes_origens.append(f"Sexta {data_origem.strftime('%d/%m/%Y')}: {n_alterados} consumos futuros corrompidos -> {situacao}.")
        
    sucesso = len(erros) == 0
    return {
        "teste": "Teste de Não-Vazamento Temporal dos Atributos da Gold",
        "sucesso": sucesso,
        "erros": erros,
        "origens_auditadas": len(origens_teste),
        "detalhes": detalhes_origens,
        "conclusao": "Atributos da Gold invariantes nas emissões testadas. Não avalia a seleção dos dias de teste." if sucesso else "FALHA DE ISOLAMENTO TEMPORAL"
    }


def executar_auditoria_completa() -> dict:
    """Executa a bateria de auditoria da camada Gold e grava o relatório."""
    pasta_auditoria = RESULTADOS / "auditoria"
    pasta_auditoria.mkdir(parents=True, exist_ok=True)
    
    fato = pd.read_parquet(SILVER / "silver_fato_diaria.parquet")
    t1 = pd.read_parquet(GOLD / "gold_t1.parquet")
    t2 = pd.read_parquet(GOLD / "gold_t2.parquet")
    
    res_estrutura = testar_estrutura_operacional(t1, t2)
    res_qualidade = testar_qualidade_dados(t1, t2)
    
    # Selecionar 5 sextas-feiras representativas de diferentes semestres para o teste de embaralhamento
    origens_comuns = sorted(set(t1["origem"]) & set(t2["origem"]))
    indices_amostra = np.linspace(5, len(origens_comuns) - 2, 5).astype(int)
    origens_teste = [origens_comuns[i] for i in indices_amostra]
    
    res_vazamento = testar_vazamento_temporal_adversarial(fato, t1, t2, origens_teste)
    
    relatorio = {
        "auditoria_realizada_em": pd.Timestamp.now().strftime("%d/%m/%Y %H:%M:%S"),
        "testes": [res_estrutura, res_qualidade, res_vazamento],
        "todos_passaram": all(t["sucesso"] for t in [res_estrutura, res_qualidade, res_vazamento])
    }
    
    with open(pasta_auditoria / "relatorio_auditoria_vazamento.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
        
    return relatorio


if __name__ == "__main__":
    rel = executar_auditoria_completa()
    print("=== Relatório de Auditoria da Camada Gold ===")
    for teste in rel["testes"]:
        status = "APROVADO [OK]" if teste["sucesso"] else "REPROVADO [FALHA]"
        print(f"\n{status} - {teste['teste']}")
        if "detalhe" in teste:
            print(f"  {teste['detalhe']}")
        if "detalhes" in teste:
            for d in teste["detalhes"]:
                print(f"  * {d}")
        if teste["erros"]:
            for e in teste["erros"]:
                print(f"  ! ERRO: {e}")

