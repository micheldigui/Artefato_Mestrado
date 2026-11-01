"""Ponto de entrada raiz para execução do Streamlit (local e Streamlit Cloud).
Pode ser executado com:
    streamlit run app.py
ou
    python executar_app.py
"""
import runpy
from pathlib import Path

if __name__ == "__main__" or "streamlit" in __name__:
    app_path = Path(__file__).resolve().parent / "src" / "app" / "main.py"
    runpy.run_path(str(app_path), run_name="__main__")
