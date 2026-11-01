import pytest
from src.core.test_zero_leakage_models import executar_auditoria_completa_vazamento


def test_zero_leakage_audit():
    rel = executar_auditoria_completa_vazamento()
    assert rel["status_geral"] == "APROVADO - ZERO DATA LEAKAGE"
    for t in rel["testes"]:
        assert t["sucesso"] is True, f"Falha no teste: {t['teste']} - Erros: {t.get('erros')}"

