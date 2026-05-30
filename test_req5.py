"""
test_req5.py — Testes unitários para REQ 5 (Adaptação Dinâmica de Humor)
Executa sem carregar dataset, spaCy ou modelos Torch (usa mocks).
"""
import sys
from collections import deque
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stubs — substituem todas as dependências pesadas
# ---------------------------------------------------------------------------

pd_stub = MagicMock()
df = MagicMock()
df.empty = False
sys.modules["pandas"] = pd_stub

sys.modules["spacy"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.feature_extraction"] = MagicMock()
sys.modules["sklearn.feature_extraction.text"] = MagicMock()
sys.modules["sklearn.metrics"] = MagicMock()
sys.modules["sklearn.metrics.pairwise"] = MagicMock()

sentiment_mock = MagicMock(return_value=[{"label": "3 stars", "score": 0.9}])
tr_stub = MagicMock()
tr_stub.pipeline.return_value = sentiment_mock
sys.modules["transformers"] = tr_stub

ld_stub = MagicMock()
ld_stub.detect.return_value = "pt"
ld_stub.DetectorFactory = MagicMock()
ld_stub.DetectorFactory.seed = 0
sys.modules["langdetect"] = ld_stub

sys.modules["deep_translator"] = MagicMock()

# ---------------------------------------------------------------------------
# Importar módulo após stubs
# ---------------------------------------------------------------------------
import importlib
engine_mod = importlib.import_module("nlp_engine")
OnePieceChatbot = engine_mod.OnePieceChatbot

# Criar instância sem __init__ (evita IO)
bot = object.__new__(OnePieceChatbot)
bot._historico_humor = deque(maxlen=20)
bot._analisador_sentimento = sentiment_mock

# ---------------------------------------------------------------------------
# Utilitário
# ---------------------------------------------------------------------------
def sep(titulo):
    print(f"\n{'=' * 54}")
    print(f"  {titulo}")
    print(f"{'=' * 54}")

# ---------------------------------------------------------------------------
# CENARIO 1 — Pergunta factual curta deve ser neutro SEM chamar o modelo
# ---------------------------------------------------------------------------
sep("Cenario 1 — Pergunta factual: 'Quem e Arlong?'")
bot._historico_humor.clear()
sentiment_mock.return_value = [{"label": "1 star", "score": 0.9}]  # modelo diria negativo
estado = bot.analisar_humor("Quem e Arlong?")
print(f"  Historico: {list(bot._historico_humor)}")
print(f"  Estado sessao: {estado}")
assert estado == "neutro", f"FALHOU: esperado neutro, obtido {estado}"
assert list(bot._historico_humor) == ["neutro"]
print("  [PASS] Forcou neutro sem chamar o modelo")

# ---------------------------------------------------------------------------
# CENARIO 2 — Marcador explícito de emoção negativa → chama modelo
# ---------------------------------------------------------------------------
sep("Cenario 2 — Marcador explicito: 'Estou estressado, me responda quem e luffy!'")
bot._historico_humor.clear()
sentiment_mock.return_value = [{"label": "1 star", "score": 0.99}]
estado = bot.analisar_humor("Estou estressado, me responda quem e luffy!")
print(f"  Historico: {list(bot._historico_humor)}")
print(f"  Estado sessao: {estado}")
assert estado == "negativo", f"FALHOU: esperado negativo, obtido {estado}"
print("  [PASS] Marcador detectado, modelo chamado, negativo registrado")

# ---------------------------------------------------------------------------
# CENARIO 3 — Frustração acumulada (3 msgs negativas → extremamente_frustrado)
# ---------------------------------------------------------------------------
sep("Cenario 3 — Frustracao acumulada (score negativo >= 0.60)")
bot._historico_humor.clear()
sentiment_mock.return_value = [{"label": "1 star", "score": 0.99}]
for msg in ["Que raiva disso!", "absurdo isso", "nao funciona nada"]:
    bot.analisar_humor(msg)
print(f"  Historico: {list(bot._historico_humor)}")
estado = bot._obter_estado_sessao()
print(f"  Estado sessao: {estado}")
assert estado == "extremamente_frustrado", f"FALHOU: obtido {estado}"
print("  [PASS] extremamente_frustrado detectado corretamente")

# ---------------------------------------------------------------------------
# CENARIO 4 — Recuperação emocional (score_neg < 0.60)
# ---------------------------------------------------------------------------
sep("Cenario 4 — Recuperacao: [negativo, negativo, positivo, positivo]")
bot._historico_humor.clear()
for humor in ["negativo", "negativo", "positivo", "positivo"]:
    bot._historico_humor.append(humor)
print(f"  Historico: {list(bot._historico_humor)}")
estado = bot._obter_estado_sessao()
print(f"  Estado sessao: {estado}")
assert estado == "positivo", f"FALHOU: obtido {estado}"
print("  [PASS] Recuperacao emocional detectada (score_neg < 0.60)")

# ---------------------------------------------------------------------------
# CENARIO 5 — reset_sessao() limpa histórico
# ---------------------------------------------------------------------------
sep("Cenario 5 — reset_sessao()")
bot._historico_humor.extend(["negativo", "negativo"])
bot.reset_sessao()
print(f"  Historico apos reset: {list(bot._historico_humor)}")
assert len(bot._historico_humor) == 0, "FALHOU: historico nao foi limpo"
print("  [PASS] reset_sessao() limpou o historico")

# ---------------------------------------------------------------------------
# Resultado final
# ---------------------------------------------------------------------------
print(f"\n{'=' * 54}")
print("  TODOS OS TESTES PASSARAM")
print(f"{'=' * 54}\n")
