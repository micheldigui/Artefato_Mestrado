import py_compile
from pathlib import Path


def test_app_compilation():
    app_main = Path(__file__).parent.parent / "src" / "app" / "main.py"
    assert app_main.exists(), "src/app/main.py deve existir"
    
    # Valida compilação de sintaxe
    res = py_compile.compile(str(app_main), doraise=True)
    assert res is not None

