"""Ponto de entrada raiz para execução do Streamlit (local e Streamlit Cloud).
Pode ser executado com:
    streamlit run app.py
ou
    python executar_app.py
"""
import os
import sys
import runpy
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
APP_DIR = RAIZ / "src" / "app"

for p in (str(RAIZ), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__" or "streamlit" in __name__:
    app_path = APP_DIR / "main.py"
    runpy.run_path(str(app_path), run_name="__main__")
