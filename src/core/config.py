"""Configurações centrais do artefato analítico-preditivo."""
from pathlib import Path
import json

# Diretórios Raiz do Projeto
RAIZ = Path(__file__).resolve().parent.parent.parent
DADOS = RAIZ / "dados"
BRONZE = DADOS / "bronze"
SILVER = DADOS / "silver"
GOLD = DADOS / "gold"
PRODUCAO = DADOS / "producao"
RESULTADOS = RAIZ / "resultados"
DOCS = RAIZ / "docs"

# Garantir existência das pastas principais
for pasta in (BRONZE, SILVER, GOLD, PRODUCAO, RESULTADOS, DOCS):
    pasta.mkdir(parents=True, exist_ok=True)

# Arquivo de configuração local (coordenadas e valores contratuais)
ARQUIVO_CONFIG_LOCAL = RAIZ / "config_local.json"

def carregar_config_local() -> dict:
    if ARQUIVO_CONFIG_LOCAL.exists():
        try:
            return json.loads(ARQUIVO_CONFIG_LOCAL.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "latitude": -23.4861,
        "longitude": -46.3483,
        "timezone": "America/Sao_Paulo",
        "valor_refeicao": {
            "2026": 14.00,
            "ate_2025": 17.36
        }
    }

CONFIG_LOCAL = carregar_config_local()

# Parâmetros de Domínio e Metodologia
ANO_INICIO_NUTRICIONISTA = 2026
PAUSA_MINIMA_DIAS_UTEIS_BLOCO = 5
SEMENTE_ALEATORIA = 42

# Horizontes Operacionais
HORIZONTES = {
    "t1": {"nome": "Preparo (Semana Seguinte)", "deslocamento_dias": 3, "min_ant": 3, "max_ant": 7},
    "t2": {"nome": "Compras (Duas Semanas)", "deslocamento_dias": 10, "min_ant": 10, "max_ant": 14}
}

# Janelas Temporais de Treinamento
JANELA_TREINO_PADRAO_MESES = 12

