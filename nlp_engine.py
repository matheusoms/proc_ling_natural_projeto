"""
nlp_engine.py — Motor NLP do Chatbot One Piece

Responsabilidades:
  - Carregamento e indexação do dataset de episódios
  - Detecção de idioma (langdetect)
  - Análise de humor com proteção contra falso-negativo em perguntas factuais
  - Rastreio de sessão emocional com média ponderada (deque)
  - Busca semântica (TF-IDF + cosseno)
  - Busca por filtros lógicos (arco, saga, tipo)
  - Tradução PT→EN com cache em memória (deep-translator)
  - Formatação localizada da resposta

Restrição acadêmica: sem LLMs generativos.
"""

import json
import logging
import re
import unicodedata
from collections import deque

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from langdetect import detect, DetectorFactory

# Garante que a detecção de idioma seja determinística
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de proteção de humor (REQ 5-A)
# ---------------------------------------------------------------------------

MARCADORES_NEGATIVOS = (
    "estou estressado", "que raiva", "absurdo", "não funciona",
    "péssimo", "odeio", "i'm angry", "so frustrated",
    "this is terrible", "que chatice", "ridículo", "impossível",
    "me deixando louco", "getting frustrated",
)

PREFIXOS_INTERROGATIVOS = (
    "quem", "qual", "quando", "onde", "como", "quanto",
    "what", "who", "when", "where", "which", "how",
    "tell me", "me diga", "me fale", "explica", "explain",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalizar_texto(texto: str) -> str:
    """
    Normaliza um texto: converte para minúsculas e remove acentuação.
    """
    if not isinstance(texto, str):
        return ""
    texto_lower = texto.lower()
    normalizado = unicodedata.normalize("NFD", texto_lower)
    sem_acento = "".join(c for c in normalizado if unicodedata.category(c) != "Mn")
    return sem_acento


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class OnePieceChatbot:
    """
    Chatbot especialista em One Piece com suporte a:
    - Busca por filtros lógicos e busca semântica via TF-IDF
    - Análise de humor com rastreio de sessão emocional (REQ 5)
    - Tradução PT→EN com cache em memória (REQ 1)
    """

    # --- Atributos de Classe (Constantes) ---
    KEYWORDS_FILTRO = [
        "filler", "semi-filler", "canon", "recap", "quantos", "listar",
        "quais", "arco", "saga", "episodios de", "mostre", "ova",
        "filme", "especial", "curta",
    ]

    KEYWORDS_TIPO = [
        "filler", "semi-filler", "canon", "recap", "ova", "filme",
        "especial", "curta-metragem", "curta", "especial tv", "especial mobile",
    ]

    MAPA_TIPO_CANONICO = {
        "filler": "Filler", "semi-filler": "Semi-Filler", "semifiller": "Semi-Filler",
        "canon": "Canon", "recap": "Recap", "ova": "OVA", "filme": "Filme",
        "especial": "Especial", "especial tv": "Especial TV",
        "especial mobile": "Especial Mobile", "curta-metragem": "Curta-Metragem",
        "curta": "Curta-Metragem",
    }
    PERSONAGENS = {
    "luffy",
    "zoro",
    "sanji",
    "nami",
    "usopp",
    "chopper",
    "robin",
    "franky",
    "brook",
    "jinbe",
    "ace",
    "sabo",
    "shanks",
    "mihawk",
    "crocodile",
    "doflamingo",
    "kaido",
    "big mom",
    "law",
    "kid",
    "buggy",
    "smoker",
    "garp",
    "sengoku",
    "whitebeard",
    "blackbeard",
    }

    ALIASES_PERSONAGENS = {
        "rufy": "luffy",
        "monkey d luffy": "luffy",
        "barba branca": "whitebeard",
        "edward newgate": "whitebeard",
        "barba negra": "blackbeard",
        "marshall d teach": "blackbeard",
        "trafalgar law": "law",
        "eustass kid": "kid",
    }
    

    def __init__(self, path: str = "dataset_episodios.json") -> None:
        # 1. Carregar dataset
        try:
            self._df = pd.read_json(path, encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Erro ao ler dataset: {e}")

        if self._df.empty:
            raise ValueError("O dataset está vazio.")

        self._df["resumo"] = self._df["resumo"].fillna("")

        # 2. Carregar modelo spaCy
        try:
            self._nlp = spacy.load("pt_core_news_lg")
        except OSError:
            raise OSError("Instale o modelo: python -m spacy download pt_core_news_lg")

        # 3. Índices e Normalizações
        self._arcos_unicos = self._df["arco"].dropna().unique().tolist()
        self._sagas_unicas = self._df["saga"].dropna().unique().tolist()
        self._arcos_normalizados = [_normalizar_texto(a) for a in self._arcos_unicos]
        self._sagas_normalizados = [_normalizar_texto(s) for s in self._sagas_unicas]

        # 4. TF-IDF com pré-processamento
        resumos_limpos = self._df["resumo"].apply(self._pre_processar_para_busca)
        self._vetorizador = TfidfVectorizer(
            min_df=2, max_df=0.8, ngram_range=(1, 2),
            strip_accents="unicode", lowercase=True
        )
        self._matriz_tfidf = self._vetorizador.fit_transform(resumos_limpos)

        # 5. IA de Sentimento
        self._analisador_sentimento = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )

        # 6. REQ 5 — Histórico emocional de sessão (deque com janela de 20)
        self._historico_humor: deque = deque(maxlen=20)

        # 7. REQ 1 — Cache de traduções PT→EN em memória
        self._cache_traducoes: dict[str, str] = {}

        # 8. REQ 1 — Inicializar tradutor (deep-translator com fallback)
        self._tradutor = self._inicializar_tradutor()

    # -----------------------------------------------------------------------
    # REQ 1 — Inicialização do tradutor
    # -----------------------------------------------------------------------

    def _inicializar_tradutor(self):
        """
        Tenta inicializar o GoogleTranslator (deep-translator).
        Retorna o objeto tradutor ou None se a biblioteca não estiver instalada.
        """
        try:
            from deep_translator import GoogleTranslator
            # Teste rápido para validar que a lib está disponível
            tradutor = GoogleTranslator(source="pt", target="en")
            logger.info("deep-translator (GoogleTranslator) carregado com sucesso.")
            return tradutor
        except ImportError:
            logger.warning(
                "deep-translator não instalado. "
                "Tradução indisponível — instale com: pip install deep-translator"
            )
            return None
        except Exception as e:
            logger.warning("Falha ao inicializar tradutor: %s", e)
            return None

    # -----------------------------------------------------------------------
    # REQ 1 — Tradução com cache (método privado)
    # -----------------------------------------------------------------------

    def _traduzir_se_necessario(self, texto: str, idioma: str) -> str:
        """
        Traduz 'texto' de PT→EN se idioma == 'en', usando cache em memória.

        Parâmetros
        ----------
        texto : str
            Texto em português a traduzir.
        idioma : str
            Idioma detectado da pergunta ('pt' ou 'en').

        Retorna
        -------
        str
            Texto traduzido (EN) ou original (PT / fallback em caso de erro).
        """
        if idioma != "en" or not texto.strip():
            return texto

        # Consultar cache
        if texto in self._cache_traducoes:
            return self._cache_traducoes[texto]

        # Sem tradutor disponível — retornar original
        if self._tradutor is None:
            return texto

        try:
            # deep-translator aceita textos de até ~5000 chars
            traduzido = self._tradutor.translate(texto[:4500])
            self._cache_traducoes[texto] = traduzido
            logger.debug("Traduzido (cache miss): '%s...'", texto[:40])
            return traduzido
        except Exception as e:
            logger.warning("Falha na tradução: %s", e)
            return texto  # fallback: texto original em PT

    # -----------------------------------------------------------------------
    # REQ 1 — Formatação localizada do episódio
    # -----------------------------------------------------------------------

    def _formatar_episodio(self, ep: dict, idioma: str = "pt") -> str:
        """
        Formata as informações de um episódio para exibição, traduzindo
        título, resumo, arco e saga quando o idioma for 'en'.

        Parâmetros
        ----------
        ep : dict
            Dicionário (ou Series) com campos do episódio.
        idioma : str
            'pt' ou 'en'.

        Retorna
        -------
        str
            Bloco de texto formatado pronto para exibição na GUI.
        """
        resumo = ep.get("resumo", "")
        if not isinstance(resumo, str) or not resumo.strip():
            resumo = (
                "Resumo não disponível." if idioma == "pt"
                else "Summary not available."
            )

        titulo = self._traduzir_se_necessario(ep.get("titulo", ""), idioma)
        resumo = self._traduzir_se_necessario(resumo, idioma)
        arco   = self._traduzir_se_necessario(ep.get("arco", ""), idioma)
        saga   = self._traduzir_se_necessario(ep.get("saga", ""), idioma)
        tipo   = ep.get("tipo", "?")

        if idioma == "en":
            return (
                f"📺 Episode {ep.get('id', '?')} — {titulo}\n"
                f"📁 {saga} > {arco} | Type: {tipo}\n"
                f"📝 {resumo}"
            )
        return (
            f"📺 Episódio {ep.get('id', '?')} — {titulo}\n"
            f"📁 {saga} > {arco} | Tipo: {tipo}\n"
            f"📝 {resumo}"
        )

    # -----------------------------------------------------------------------
    # MÉTODOS DE SUPORTE
    # -----------------------------------------------------------------------

    def _pre_processar_para_busca(self, texto: str) -> str:
        texto_norm = _normalizar_texto(texto)
        doc = self._nlp(texto_norm)
        tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and len(t.text) > 1]
        return " ".join(tokens)

    def detectar_idioma(self, texto: str) -> str:
        try:
            return "en" if detect(texto) == "en" else "pt"
        except Exception:
            return "pt"
    
    def _aplicar_aliases(self, texto: str) -> str:
        texto = _normalizar_texto(texto)

        for alias, canonico in self.ALIASES_PERSONAGENS.items():
            texto = texto.replace(
                _normalizar_texto(alias),
                canonico
            )

            return texto

    def _extrair_personagens(self, texto: str) -> list[str]:
        texto = self._aplicar_aliases(texto)
        encontrados = []
        for personagem in self.PERSONAGENS:
            padrao = rf"\b{re.escape(personagem)}\b"
            if re.search(padrao, texto):
                encontrados.append(personagem)
        return encontrados
    # -----------------------------------------------------------------------
    # REQ 5 — Classificação de humor com proteção contra falso-negativo
    # -----------------------------------------------------------------------

    def _classificar_humor_imediato(self, texto: str) -> str:
        """
        Chama o pipeline nlptown para classificar o sentimento bruto do texto.
        Retorna 'positivo', 'neutro' ou 'negativo'.
        """
        try:
            res = self._analisador_sentimento(texto[:512])[0]
            label = res["label"]
            if label in ("1 star", "2 stars"):
                return "negativo"
            if label == "3 stars":
                return "neutro"
            return "positivo"
        except Exception:
            return "neutro"

    def _obter_estado_sessao(self) -> str:
        """
        Calcula o estado emocional da sessão usando média ponderada das
        últimas 5 mensagens do histórico.

        Retorna
        -------
        str
            'extremamente_frustrado' se score_negativo >= 0.60,
            caso contrário o humor da última mensagem.
        """
        if len(self._historico_humor) < 3:
            return (
                self._historico_humor[-1]
                if self._historico_humor else "neutro"
            )

        janela = list(self._historico_humor)[-5:]
        pesos = list(range(1, len(janela) + 1))
        soma_pesos = sum(pesos)
        score_neg = sum(
            p for p, e in zip(pesos, janela) if e == "negativo"
        ) / soma_pesos

        if score_neg >= 0.60:
            return "extremamente_frustrado"
        return self._historico_humor[-1]

    def analisar_humor(self, texto: str) -> str:
        """
        Analisa o humor da mensagem com proteção contra falso-negativo
        em perguntas factuais curtas (REQ 5-A).

        Perguntas factuais curtas (≤10 palavras, sem marcadores negativos
        explícitos) são classificadas como 'neutro' sem chamar o modelo.

        Atualiza self._historico_humor e retorna o estado de sessão.
        """
        texto_lower = texto.lower().strip()

        eh_interrogativa = any(
            texto_lower.startswith(p) for p in PREFIXOS_INTERROGATIVOS
        )
        tem_marcador = any(m in texto_lower for m in MARCADORES_NEGATIVOS)
        eh_curta = len(texto.split()) <= 10

        if eh_interrogativa and eh_curta and not tem_marcador:
            # Pergunta factual neutra — não chamar o modelo (evita falso-negativo)
            self._historico_humor.append("neutro")
            return self._obter_estado_sessao()

        # Caso geral: classificar via pipeline
        estado = self._classificar_humor_imediato(texto)
        self._historico_humor.append(estado)
        return self._obter_estado_sessao()

    # -----------------------------------------------------------------------
    # REQ 5-C — Reset de sessão
    # -----------------------------------------------------------------------

    def reset_sessao(self) -> None:
        """
        Limpa o histórico emocional da sessão e o cache de traduções.
        Deve ser chamado pelo botão 'Nova Conversa' na GUI.
        """
        self._historico_humor.clear()
        logger.info("Histórico emocional de sessão reiniciado.")

    # -----------------------------------------------------------------------
    # MÉTODOS DE PROCESSAMENTO
    # -----------------------------------------------------------------------

    def processar_pergunta(self, texto: str) -> dict:
        texto_norm = _normalizar_texto(texto)
        return {
            "intencao": self._detectar_intencao(texto_norm),
            "entidades": self._extrair_entidades(texto, texto_norm),
            "texto_normalizado": texto_norm,
        }

    def gerar_resposta(self, texto_usuario: str) -> str:
        """
        Ponto de entrada principal: detecta idioma + humor, executa a busca
        e retorna a resposta formatada e localizada (PT ou EN).
        """
        idioma = self.detectar_idioma(texto_usuario)
        humor  = self.analisar_humor(texto_usuario)   # atualiza histórico
        analise = self.processar_pergunta(texto_usuario)

        if analise["intencao"] == "filtro":
            resposta_base = self._busca_por_filtros(analise["entidades"], idioma)
        else:
            resposta_base = self._busca_semantica(texto_usuario, idioma)

        # --- Prefixos empáticos bilíngues com suporte a 'extremamente_frustrado' ---
        prefixos: dict[str, dict[str, str]] = {
            "pt": {
                "positivo": "Que bom ver sua animação! ✨",
                "negativo": "Sinto muito pela frustração. Deixa eu te ajudar: ⚓",
                "neutro":   "Encontrei isso para você:",
                "extremamente_frustrado": (
                    "Ei, percebo que você está bem frustrado... "
                    "Respira fundo, vou te ajudar! 🏴‍☠️"
                ),
            },
            "en": {
                "positivo": "I love the enthusiasm! ✨",
                "negativo": "I'm sorry you feel that way. Let me help: ⚓",
                "neutro":   "Here is what I found:",
                "extremamente_frustrado": (
                    "Hey, I can see you're really frustrated... "
                    "Take a breath, I'm here to help! 🏴‍☠️"
                ),
            },
        }

        pre = prefixos[idioma].get(humor, prefixos[idioma]["neutro"])

        # Mensagem de "não encontrado" localizada
        if idioma == "en" and "Não encontrei" in resposta_base:
            resposta_base = "I couldn't find any results for your search."

        return f"{pre}\n\n{resposta_base}"

    def listar_arcos(self) -> str:
        linhas = ["📚 **Arcos disponíveis no dataset:**\n"]
        for saga in self._sagas_unicas:
            arcos = self._df[self._df["saga"] == saga]["arco"].dropna().unique()
            linhas.append(f"🗺️  **{saga}**")
            for arco in arcos:
                total = len(self._df[self._df["arco"] == arco])
                linhas.append(f"    • {arco} ({total} episódios)")
            linhas.append("")
        return "\n".join(linhas)

    # -----------------------------------------------------------------------
    # MÉTODOS PRIVADOS
    # -----------------------------------------------------------------------

    def _detectar_intencao(self, texto_norm: str) -> str:
        for kw in self.KEYWORDS_FILTRO:
            if _normalizar_texto(kw) in texto_norm:
                return "filtro"
        return "semantica"

    def _extrair_entidades(self, texto_orig: str, texto_norm: str) -> dict:
        arco, saga, tipo = None, None, None

        for idx, a_norm in enumerate(self._arcos_normalizados):
            if a_norm.replace("arco ", "") in texto_norm or a_norm in texto_norm:
                arco = self._arcos_unicos[idx]
                break

        for idx, s_norm in enumerate(self._sagas_normalizados):
            if s_norm.replace("saga ", "") in texto_norm or s_norm in texto_norm:
                saga = self._sagas_unicas[idx]
                break

        for kw in self.KEYWORDS_TIPO:
            kn = _normalizar_texto(kw)
            if kn in texto_norm:
                tipo = self.MAPA_TIPO_CANONICO.get(kn, kw.capitalize())
                break

        return {"arco": arco, "saga": saga, "tipo": tipo, "personagens": self._extrair_personagens(texto_orig)}

    def _busca_por_filtros(self, ent: dict, idioma: str = "pt") -> str:
        mask = pd.Series([True] * len(self._df), index=self._df.index)
        filtros = []

        if ent["arco"]:
            mask &= self._df["arco"] == ent["arco"]
            filtros.append(f"arco='{ent['arco']}'")
        if ent["saga"] and not ent["arco"]:
            mask &= self._df["saga"] == ent["saga"]
            filtros.append(f"saga='{ent['saga']}'")
        if ent["tipo"]:
            mask &= self._df["tipo"].str.lower() == ent["tipo"].lower()
            filtros.append(f"tipo='{ent['tipo']}'")

        res = self._df[mask]

        if not filtros:
            counts = self._df["tipo"].value_counts()
            res_txt = ["**Estatísticas gerais:**\n"]
            for t, q in counts.items():
                res_txt.append(f"  • {t}: {q}")
            return "\n".join(res_txt)

        if res.empty:
            return (
                "🔍 Nada encontrado para esses filtros."
                if idioma == "pt"
                else "🔍 Nothing found for these filters."
            )

        linhas = [f"🔎 Filtro: {' | '.join(filtros)}", f"📋 Total: {len(res)}\n"]
        for _, ep in res.head(10).iterrows():
            linhas.append(f"  Ep. {ep['id']} — {ep['titulo']} ({ep['tipo']})")
        return "\n".join(linhas)

    def _busca_semantica(self, texto: str, idioma: str = "pt") -> str:

        t_limpo = self._pre_processar_para_busca(texto)

        v_pergunta = self._vetorizador.transform([t_limpo])

        sim = cosine_similarity(
            v_pergunta,
            self._matriz_tfidf
        ).flatten()

        # Detecta personagens mencionados na pergunta
        personagens = self._extrair_personagens(texto)

        # Dá bônus para episódios que mencionam esses personagens
        for personagem in personagens:

            mask_resumo = self._df["resumo"].str.contains(
                personagem,
                case=False,
                na=False
            )

            mask_titulo = self._df["titulo"].str.contains(
                personagem,
                case=False,
                na=False
            )

            sim[mask_resumo] += 0.25
            sim[mask_titulo] += 0.40

        idx = int(sim.argmax())

        if sim[idx] < 0.15:
            return (
                "Não encontrei nada específico. Tente citar um personagem ou luta!"
                if idioma == "pt"
                else "I couldn't find anything specific. Try mentioning a character or fight!"
            )

        ep = self._df.iloc[idx].to_dict()

        return self._formatar_episodio(ep, idioma)