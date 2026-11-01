"""Torneio de Modelos Preditivos em Validação Walk-Forward Estrita (Zero Data Leakage).

Implementa a taxonomia completa de métodos de Machine Learning e Ensembles:
  1. Baselines Estatísticos: Nível Recente (10d) e Mesmo Dia da Semana.
  2. Modelos Lineares Regularizados: Regressão Linear, Ridge, Lasso, ElasticNet.
  3. Ensembles de Bagging: Random Forest e ExtraTrees.
  4. Ensembles de Boosting: Gradient Boosting, AdaBoost, LightGBM, XGBoost e CatBoost.
  5. Ensembles de Voting: Votação com média ponderada de modelos heterogêneos.
  6. Ensembles de Stacking: Meta-aprendizado com StackingRegressor.
  7. Ensembles de Blending: Validação holdout temporal com pesos não-negativos (NNLS).
  8. Ensembles de Cascading: Encadeamento em dois estágios (âncora linear -> resíduo não-linear).

Ablation Study em 4 Cenários de Informação nos Horizontes T+1 (Preparo) e T+2 (Compras).
"""
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    VotingRegressor,
    StackingRegressor
)
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from joblib import Parallel, delayed

from .config import GOLD, HORIZONTES, RESULTADOS, SEMENTE_ALEATORIA

PASTA_MODELOS = RESULTADOS / "modelos"
PASTA_MODELOS.mkdir(parents=True, exist_ok=True)

LIMITES_ALVO = (0.2, 2.5)  # Limiares de segurança para a razão relativa de consumo


# =====================================================================
# 1. Classes Customizadas de Ensembles: Blending e Cascading
# =====================================================================

class BlendingRegressor(BaseEstimator, RegressorMixin):
    """Ensemble de Blending: divide o treino em base e holdout temporal recente.
    
    Ajusta modelos base no conjunto principal e aprende os pesos ótimos não-negativos
    (via Non-Negative Least Squares - NNLS) sobre o holdout temporal.
    """
    def __init__(self, val_split: float = 0.2):
        self.val_split = val_split

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n = len(X_arr)
        n_train = max(int(n * (1 - self.val_split)), 15)
        
        X_tr, y_tr = X_arr[:n_train], y_arr[:n_train]
        X_val, y_val = X_arr[n_train:], y_arr[n_train:]
        
        self.m1_ = Ridge(alpha=20.0, random_state=SEMENTE_ALEATORIA)
        self.m2_ = ExtraTreesRegressor(n_estimators=60, min_samples_leaf=5, random_state=SEMENTE_ALEATORIA, n_jobs=1)
        self.m3_ = CatBoostRegressor(iterations=60, learning_rate=0.06, depth=4, verbose=0, random_seed=SEMENTE_ALEATORIA, thread_count=1)
        
        self.m1_.fit(X_tr, y_tr)
        self.m2_.fit(X_tr, y_tr)
        self.m3_.fit(X_tr, y_tr)
        
        if len(X_val) >= 5:
            p1 = self.m1_.predict(X_val)
            p2 = self.m2_.predict(X_val)
            p3 = self.m3_.predict(X_val)
            P = np.column_stack([p1, p2, p3])
            w, _ = nnls(P, y_val)
            total_w = np.sum(w)
            self.weights_ = w / total_w if total_w > 0 else np.array([1/3, 1/3, 1/3])
        else:
            self.weights_ = np.array([0.4, 0.3, 0.3])
            
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        p1 = self.m1_.predict(X_arr)
        p2 = self.m2_.predict(X_arr)
        p3 = self.m3_.predict(X_arr)
        P = np.column_stack([p1, p2, p3])
        return P @ self.weights_


class CascadingRegressor(BaseEstimator, RegressorMixin):
    """Ensemble de Cascading (Dois Estágios):
    
    Estágio 1: Modelo linear regularizado captura a linha de base e tendências globais.
    Estágio 2: Modelo Gradient Boosting aprende a modelar os resíduos não-lineares,
               utilizando as variáveis originais concatenadas com a predição do 1º estágio.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        
        # 1º Estágio: Tendência Global
        self.stage1_ = Ridge(alpha=30.0, random_state=SEMENTE_ALEATORIA)
        self.stage1_.fit(X_arr, y_arr)
        pred_stage1 = self.stage1_.predict(X_arr)
        
        # 2º Estágio: Aprendizado nos Resíduos
        residuos = y_arr - pred_stage1
        X_estagio2 = np.column_stack([X_arr, pred_stage1])
        self.stage2_ = GradientBoostingRegressor(
            loss="absolute_error",
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=8,
            random_state=SEMENTE_ALEATORIA
        )
        self.stage2_.fit(X_estagio2, residuos)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        pred_stage1 = self.stage1_.predict(X_arr)
        X_estagio2 = np.column_stack([X_arr, pred_stage1])
        residuos_pred = self.stage2_.predict(X_estagio2)
        return pred_stage1 + residuos_pred


# =====================================================================
# 2. Catálogo de Modelos e Pipelines de Transformação
# =====================================================================

def criar_pipeline_linear(estimador) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        estimador
    )


def criar_pipeline_arvore(estimador) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        estimador
    )


def obter_catalogo_modelos() -> Dict[str, dict]:
    """Retorna o catálogo completo de algoritmos organizados por família conceitual."""
    modelos = {}

    # --- Família 1: Modelos Lineares Regularizados ---
    modelos["Regressao_Linear"] = {
        "familia": "Linear",
        "tipo_ensemble": "Nenhum (Modelo Base)",
        "construtor": lambda: criar_pipeline_linear(LinearRegression())
    }
    modelos["Ridge"] = {
        "familia": "Linear Regularizado",
        "tipo_ensemble": "Nenhum (L2 Regularized)",
        "construtor": lambda: criar_pipeline_linear(Ridge(alpha=20.0, random_state=SEMENTE_ALEATORIA))
    }
    modelos["Lasso"] = {
        "familia": "Linear Regularizado",
        "tipo_ensemble": "Nenhum (L1 Sparse)",
        "construtor": lambda: criar_pipeline_linear(Lasso(alpha=0.01, max_iter=3000, random_state=SEMENTE_ALEATORIA))
    }
    modelos["ElasticNet"] = {
        "familia": "Linear Regularizado",
        "tipo_ensemble": "Nenhum (L1+L2 Regularized)",
        "construtor": lambda: criar_pipeline_linear(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=3000, random_state=SEMENTE_ALEATORIA))
    }

    # --- Família 2: Ensembles de Bagging (Bootstrap Aggregating) ---
    modelos["Ensemble_Bagging_RandomForest"] = {
        "familia": "Ensemble",
        "tipo_ensemble": "Bagging (Bootstrap Paralelo)",
        "construtor": lambda: criar_pipeline_arvore(RandomForestRegressor(
            n_estimators=80, min_samples_leaf=5, max_features=0.6, random_state=SEMENTE_ALEATORIA, n_jobs=1
        ))
    }
    modelos["Ensemble_Bagging_ExtraTrees"] = {
        "familia": "Ensemble",
        "tipo_ensemble": "Bagging (Árvores Extremamente Aleatórias)",
        "construtor": lambda: criar_pipeline_arvore(ExtraTreesRegressor(
            n_estimators=80, min_samples_leaf=5, max_features=0.6, random_state=SEMENTE_ALEATORIA, n_jobs=1
        ))
    }

    # --- Família 3: Ensembles de Boosting (Correção Sequencial de Resíduos) ---
    modelos["Ensemble_Boosting_Gradient"] = {
        "familia": "Ensemble",
        "tipo_ensemble": "Boosting (Gradient Boosting Robusto)",
        "construtor": lambda: criar_pipeline_arvore(GradientBoostingRegressor(
            loss="absolute_error", n_estimators=80, learning_rate=0.06, max_depth=3, min_samples_leaf=8, random_state=SEMENTE_ALEATORIA
        ))
    }
    modelos["Ensemble_Boosting_AdaBoost"] = {
        "familia": "Ensemble",
        "tipo_ensemble": "Boosting (AdaBoost Adaptativo)",
        "construtor": lambda: criar_pipeline_arvore(AdaBoostRegressor(
            n_estimators=60, learning_rate=0.06, loss="linear", random_state=SEMENTE_ALEATORIA
        ))
    }
    modelos["Ensemble_Boosting_LightGBM"] = {
        "familia": "Ensemble",
        "tipo_ensemble": "Boosting (LightGBM Leaf-wise)",
        "construtor": lambda: LGBMRegressor(
            objective="l1", n_estimators=80, learning_rate=0.06, max_depth=3, num_leaves=7,
            min_child_samples=15, subsample=0.8, colsample_bytree=0.8, random_state=SEMENTE_ALEATORIA, verbose=-1, n_jobs=1
        )
    }
    modelos["Ensemble_Boosting_XGBoost"] = {
        "familia": "Ensemble",
        "tipo_ensemble": "Boosting (XGBoost Regularizado)",
        "construtor": lambda: XGBRegressor(
            objective="reg:absoluteerror", n_estimators=80, learning_rate=0.06, max_depth=3,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=SEMENTE_ALEATORIA, verbosity=0, n_jobs=1
        )
    }
    modelos["Ensemble_Boosting_CatBoost"] = {
        "familia": "Ensemble",
        "tipo_ensemble": "Boosting (CatBoost Simétrico)",
        "construtor": lambda: CatBoostRegressor(
            loss_function="MAE", iterations=80, learning_rate=0.06, depth=4,
            random_seed=SEMENTE_ALEATORIA, verbose=0, thread_count=1
        )
    }

    # --- Família 4: Ensembles de Voting (Votação / Média Ponderada) ---
    def construir_voting():
        est = [
            ("ridge", criar_pipeline_linear(Ridge(alpha=20.0, random_state=SEMENTE_ALEATORIA))),
            ("extratrees", criar_pipeline_arvore(ExtraTreesRegressor(n_estimators=60, min_samples_leaf=5, random_state=SEMENTE_ALEATORIA, n_jobs=1))),
            ("catboost", CatBoostRegressor(iterations=60, learning_rate=0.06, depth=4, verbose=0, random_seed=SEMENTE_ALEATORIA, thread_count=1))
        ]
        return VotingRegressor(estimators=est, weights=[0.4, 0.3, 0.3])

    modelos["Ensemble_Voting_Heterogeneo"] = {
        "familia": "Ensemble",
        "tipo_ensemble": "Voting (Votação Ponderada Heterogênea)",
        "construtor": construir_voting
    }

    # --- Família 5: Ensembles de Stacking (Meta-Aprendizado) ---
    def construir_stacking():
        base_est = [
            ("ridge", criar_pipeline_linear(Ridge(alpha=20.0, random_state=SEMENTE_ALEATORIA))),
            ("extratrees", criar_pipeline_arvore(ExtraTreesRegressor(n_estimators=40, min_samples_leaf=5, random_state=SEMENTE_ALEATORIA, n_jobs=1))),
            ("catboost", CatBoostRegressor(iterations=40, learning_rate=0.08, depth=3, verbose=0, random_seed=SEMENTE_ALEATORIA, thread_count=1))
        ]
        meta = Ridge(alpha=10.0, positive=True, random_state=SEMENTE_ALEATORIA)
        return StackingRegressor(estimators=base_est, final_estimator=meta, cv=3, n_jobs=1)

    modelos["Ensemble_Stacking_MetaModel"] = {
        "familia": "Ensemble",
        "tipo_ensemble": "Stacking (Meta-Regressão Não-Negativa)",
        "construtor": construir_stacking
    }

    # --- Família 6: Ensembles de Blending (Holdout Temporal) ---
    modelos["Ensemble_Blending_Holdout"] = {
        "familia": "Ensemble",
        "tipo_ensemble": "Blending (Holdout Temporal NNLS)",
        "construtor": lambda: criar_pipeline_arvore(BlendingRegressor(val_split=0.2))
    }

    # --- Família 7: Ensembles de Cascading (Encadeamento em Dois Estágios) ---
    modelos["Ensemble_Cascading_Estagios"] = {
        "familia": "Ensemble",
        "tipo_ensemble": "Cascading (Dois Estágios Sequenciais)",
        "construtor": lambda: criar_pipeline_arvore(CascadingRegressor())
    }

    return modelos


# =====================================================================
# 3. Definição dos 4 Cenários de Informação (Ablation Study)
# =====================================================================

def definir_cenarios_features() -> Dict[str, List[str]]:
    """Define as variáveis pertencentes a cada cenário experimental de informação."""
    f_nivel = [
        "nivel_base", "nivel_10d", "nivel_20d", "nivel_mesmo_dia", "nivel_exponencial",
        "media_5d", "mediana_5d", "media_10d", "media_20d", "ultimo_consumo",
        "mesmo_dia_ha_1_ocorrencias", "mesmo_dia_ha_2_ocorrencias", "mesmo_dia_ha_3_ocorrencias",
        "tendencia_nivel", "dispersao_recente", "coef_variacao_10d",
        "rel_ultimo_consumo", "rel_media_5d", "rel_nivel_exponencial", "rel_mesmo_dia_ha_1_ocorrencias"
    ]
    f_calendario = [
        "dia_seg", "dia_ter", "dia_qua", "dia_qui", "dia_sex",
        "dias_ate_fim_bloco", "dias_desde_inicio_bloco", "semana_do_bloco", "fim_de_bloco", "inicio_de_bloco",
        "vespera_de_pausa", "pos_pausa", "dias_sem_aula_antes", "dias_letivos_na_semana",
        "efeito_posicao_bloco", "mesma_epoca_ano_passado", "efeito_evento"
    ]
    f_cardapio = [
        "efeito_proteina", "efeito_preparo", "efeito_prato", "sem_cardapio"
    ]
    f_clima = [
        "previsao_chuva_mm", "previsao_chuva_manha_mm", "previsao_chuva_acum3d_mm", "previsao_temp_max"
    ]
    f_plano = [
        "rel_plano_humano"
    ]

    c1 = f_nivel + f_calendario
    c2 = c1 + f_cardapio
    c3 = c2 + f_clima
    c4 = c3 + f_plano

    return {
        "Cenario_1_Base_Cardapio": c1,
        "Cenario_2_Com_Cardapio": c2,
        "Cenario_3_Com_Clima": c3,
        "Cenario_4_Com_Nutricionista": c4
    }


def arredondar_refeicoes(valores: np.ndarray) -> np.ndarray:
    """Arredonda para refeição inteira: de 0.5 para cima, arredonda para o inteiro superior."""
    return np.floor(np.asarray(valores, dtype=float) + 0.5)


def calcular_metricas_operacionais(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula métricas estritamente operacionais de erro e balanço sobra/déficit."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    erros = y_pred - y_real
    erros_abs = np.abs(erros)
    
    mae = float(np.mean(erros_abs))
    rmse = float(np.sqrt(np.mean(erros ** 2)))
    
    # R2 Score
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    ss_res = np.sum(erros ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    # WAPE (%)
    soma_real = float(np.sum(y_real))
    wape = float(np.sum(erros_abs) / soma_real * 100.0) if soma_real > 0 else 0.0
    
    # Sobra e Déficit Potencial
    sobra_total = int(np.sum(np.maximum(0, erros)))
    deficit_total = int(np.sum(np.maximum(0, -erros)))
    
    taxa_dias_sobra = float(np.mean(erros > 0) * 100.0)
    taxa_dias_deficit = float(np.mean(erros < 0) * 100.0)
    
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3),
        "WAPE_pct": round(wape, 2),
        "Total_Sobra_ref": sobra_total,
        "Total_Deficit_ref": deficit_total,
        "Taxa_Dias_Sobra_pct": round(taxa_dias_sobra, 1),
        "Taxa_Dias_Deficit_pct": round(taxa_dias_deficit, 1)
    }


# =====================================================================
# 4. Motor de Validação Walk-Forward Estrita (12 Meses de Treino)
# =====================================================================

def avaliar_um_algoritmo(
    nome_alg: str,
    info_alg: dict,
    horizonte: str,
    nome_cenario: str,
    colunas_validas: list,
    gold: pd.DataFrame,
    origens_2026: list
) -> Tuple[dict, list]:
    """Treina e avalia um algoritmo específico através de Walk-Forward temporal em 2026."""
    y_reais = []
    y_preds = []
    datas_pred = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for origem in origens_2026:
            alvo_semana = gold[(gold["origem"] == origem) & gold["consumo"].notna() & gold["usar_no_treino"]]
            if alvo_semana.empty:
                continue
                
            data_limite_inicio = pd.to_datetime(origem) - pd.DateOffset(months=12)
            treino = gold[
                (gold["data"] < origem) &
                (gold["data"] >= data_limite_inicio) &
                gold["usar_no_treino"] &
                gold["consumo"].notna() &
                gold["alvo_relativo"].notna()
            ].copy()
            
            if len(treino) < 20:
                continue
                
            X_treino = treino[colunas_validas].copy()
            y_treino = treino["alvo_relativo"].clip(*LIMITES_ALVO).copy()
            
            modelo = info_alg["construtor"]()
            modelo.fit(X_treino, y_treino)
            
            X_alvo = alvo_semana[colunas_validas].copy()
            pred_relativa = np.clip(modelo.predict(X_alvo), LIMITES_ALVO[0], LIMITES_ALVO[1])
            pred_refeicoes = arredondar_refeicoes(pred_relativa * alvo_semana["nivel_base"].values)
            
            y_reais.extend(alvo_semana["consumo"].values)
            y_preds.extend(pred_refeicoes)
            datas_pred.extend(alvo_semana["data"].values)
            
    if not y_reais:
        return None, []
        
    m = calcular_metricas_operacionais(y_reais, y_preds)
    metrica_dict = {
        "horizonte": horizonte,
        "cenario": nome_cenario,
        "algoritmo": nome_alg,
        "familia": info_alg["familia"],
        "tipo_ensemble": info_alg["tipo_ensemble"],
        "dias_avaliados": len(y_reais),
        **m
    }
    preds_list = [
        {"horizonte": horizonte, "cenario": nome_cenario, "algoritmo": nome_alg, "data": dt, "consumo_real": real, "previsao": p}
        for dt, real, p in zip(datas_pred, y_reais, y_preds)
    ]
    return metrica_dict, preds_list


def executar_torneio_walk_forward() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executa a validação Walk-Forward para todos os modelos, cenários e horizontes."""
    catalogo_modelos = obter_catalogo_modelos()
    cenarios = definir_cenarios_features()
    
    linhas_metricas = []
    linhas_predicoes = []
    
    print("[*] Iniciando Torneio Walk-Forward (Isolamento Temporal Estrito de 12 Meses)...")

    for horizonte in ("t1", "t2"):
        caminho_gold = GOLD / f"gold_{horizonte}.parquet"
        gold = pd.read_parquet(caminho_gold)
        
        # Cria a feature 'rel_plano_humano' em T+1
        if horizonte == "t1":
            plano_raw = gold["plano_nutricionista"].fillna(gold["reserva_discente"])
            gold["rel_plano_humano"] = (plano_raw / gold["nivel_base"]).fillna(1.0)
            cenarios_ativos = cenarios
        else:
            # Em T+2, o Cenário 4 (com nutricionista) é proibido por design
            cenarios_ativos = {k: v for k, v in cenarios.items() if k != "Cenario_4_Com_Nutricionista"}
            
        # Avaliação prospectiva ao longo de 2026
        gold_2026 = gold[gold["ano"] == 2026].copy()
        origens_2026 = sorted(gold_2026["origem"].unique())
        
        print(f"\n--> Horizonte [{horizonte.upper()}]: {len(origens_2026)} semanas-teste de 2026...")
        
        for nome_cenario, colunas_cenario in cenarios_ativos.items():
            print(f"    * Executando {nome_cenario}...")
            colunas_validas = [c for c in colunas_cenario if c in gold.columns]
            
            # --- 1. Avaliar Baselines Estatísticos ---
            for nome_base, col_base in [("Baseline_Nivel_Recente_10d", "nivel_10d"), ("Baseline_Mesmo_Dia_Semana", "nivel_mesmo_dia")]:
                df_teste = gold_2026[gold_2026["consumo"].notna() & gold_2026["usar_no_treino"]].copy()
                pred_base = arredondar_refeicoes(df_teste[col_base].fillna(df_teste["nivel_base"]).values)
                
                m = calcular_metricas_operacionais(df_teste["consumo"].values, pred_base)
                linhas_metricas.append({
                    "horizonte": horizonte,
                    "cenario": nome_cenario,
                    "algoritmo": nome_base,
                    "familia": "Baseline Estatístico",
                    "tipo_ensemble": "Nenhum (Heurística Estática)",
                    "dias_avaliados": len(df_teste),
                    **m
                })
                
                for dt, real, p in zip(df_teste["data"], df_teste["consumo"], pred_base):
                    linhas_predicoes.append({
                        "horizonte": horizonte, "cenario": nome_cenario, "algoritmo": nome_base,
                        "data": dt, "consumo_real": real, "previsao": p
                    })

            # --- 2. Avaliar Algoritmos de Machine Learning e Ensembles ---
            for nome_alg, info_alg in catalogo_modelos.items():
                m_dict, preds_list = avaliar_um_algoritmo(
                    nome_alg, info_alg, horizonte, nome_cenario, colunas_validas, gold, origens_2026
                )
                if m_dict is not None:
                    linhas_metricas.append(m_dict)
                    linhas_predicoes.extend(preds_list)
                    print(f"        -> {nome_alg}: MAE={m_dict['MAE']} | WAPE={m_dict['WAPE_pct']}%", flush=True)
                        
    df_metricas = pd.DataFrame(linhas_metricas)
    df_predicoes = pd.DataFrame(linhas_predicoes)
    
    # Salvar resultados
    df_metricas.to_csv(PASTA_MODELOS / "metricas_torneio_modelos.csv", sep=";", index=False, encoding="utf-8-sig")
    df_metricas.to_parquet(PASTA_MODELOS / "metricas_torneio_modelos.parquet", index=False)
    df_predicoes.to_parquet(PASTA_MODELOS / "predicoes_detalhadas_torneio.parquet", index=False)
    
    return df_metricas, df_predicoes


def exibir_tabelas_resumo(df_metricas: pd.DataFrame) -> None:
    """Exibe rankings sumarizados por horizonte e cenário."""
    print("\n=== TOP 5 MODELOS NO HORIZONTE T+1 (Preparo Diário) ===")
    t1_df = df_metricas[df_metricas["horizonte"] == "t1"].sort_values("MAE")
    print(t1_df[["cenario", "algoritmo", "tipo_ensemble", "MAE", "RMSE", "WAPE_pct", "Total_Sobra_ref", "Total_Deficit_ref"]].head(8).to_string(index=False))
    
    print("\n=== TOP 5 MODELOS NO HORIZONTE T+2 (Compras Quinzenais - 100% Autônomo) ===")
    t2_df = df_metricas[df_metricas["horizonte"] == "t2"].sort_values("MAE")
    print(t2_df[["cenario", "algoritmo", "tipo_ensemble", "MAE", "RMSE", "WAPE_pct", "Total_Sobra_ref", "Total_Deficit_ref"]].head(8).to_string(index=False))


if __name__ == "__main__":
    df_m, df_p = executar_torneio_walk_forward()
    exibir_tabelas_resumo(df_m)

