"""Iniciador Prático da Aplicação Web do Artefato Preditivo.
Execute: python executar_app.py
"""
import subprocess
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).parent / "src" / "app" / "main.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print("=" * 70)
    print("Iniciando Aplicação Web Streamlit do Artefato Analítico-Preditivo...")
    print(f"Comando: {' '.join(cmd)}")
    print("=" * 70)
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
