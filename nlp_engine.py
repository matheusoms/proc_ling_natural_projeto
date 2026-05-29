
import json
import re
import unicodedata

import pandas as pd  # <--- ESSA LINHA É A QUE ESTÁ FALTANDO!
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline
from langdetect import detect, DetectorFactory

# Garante que a detecção de idioma seja determinística
DetectorFactory.seed = 0

def _normalizar_texto(texto: str) -> str:
    """
    Normaliza um texto: converte para minúsculas e remove acentuação.
    """
    if not isinstance(texto, str):
        return ""
    texto_lower = texto.lower()
    # Decompor unicode e remover diacríticos (acentos)
    normalizado = unicodedata.normalize("NFD", texto_lower)
    sem_acento = "".join(c for c in normalizado if unicodedata.category(c) != "Mn")
    return sem_acento
class OnePieceChatbot:
    """
    Chatbot especialista em One Piece com suporte a busca por filtros lógicos
    e busca semântica via TF-IDF + similaridade de cossenos.
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

        # 3. Índices e Normalizações (IMPORTANTE: Criar antes do TF-IDF)
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

    # --- MÉTODOS DE SUPORTE ---

    def _pre_processar_para_busca(self, texto: str) -> str:
        texto_norm = _normalizar_texto(texto)
        doc = self._nlp(texto_norm)
        tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and len(t.text) > 1]
        return " ".join(tokens)

    def detectar_idioma(self, texto: str) -> str:
        try:
            return "en" if detect(texto) == "en" else "pt"
        except:
            return "pt"

    def analisar_humor(self, texto: str) -> str:
        try:
            res = self._analisador_sentimento(texto[:512])[0]
            label = res['label']
            if label in ['1 star', '2 stars']: return "negativo"
            if label == '3 stars': return "neutro"
            return "positivo"
        except:
            return "neutro"

    # --- MÉTODOS DE PROCESSAMENTO ---

    def processar_pergunta(self, texto: str) -> dict:
        texto_norm = _normalizar_texto(texto)
        return {
            "intencao": self._detectar_intencao(texto_norm),
            "entidades": self._extrair_entidades(texto, texto_norm),
            "texto_normalizado": texto_norm,
        }

    def gerar_resposta(self, texto_usuario: str) -> str:
        idioma = self.detectar_idioma(texto_usuario)
        humor = self.analisar_humor(texto_usuario)
        analise = self.processar_pergunta(texto_usuario)
        
        if analise["intencao"] == "filtro":
            resposta_base = self._busca_por_filtros(analise["entidades"])
        else:
            resposta_base = self._busca_semantica(texto_usuario)

        prefixos = {
            "pt": {
                "positivo": "Que bom ver sua animação! ✨",
                "negativo": "Sinto muito pela frustração. Deixa eu te ajudar: ⚓",
                "neutro": "Encontrei isso para você:"
            },
            "en": {
                "positivo": "I love the enthusiasm! ✨",
                "negativo": "I'm sorry you feel that way. Let me help: ⚓",
                "neutro": "Here is what I found:"
            }
        }
        
        pre = prefixos[idioma][humor]
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

    # --- MÉTODOS PRIVADOS ---

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

        return {"arco": arco, "saga": saga, "tipo": tipo}

    def _busca_por_filtros(self, ent: dict) -> str:
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
            for t, q in counts.items(): res_txt.append(f"  • {t}: {q}")
            return "\n".join(res_txt)

        if res.empty: return "🔍 Nada encontrado para esses filtros."

        linhas = [f"🔎 Filtro: {' | '.join(filtros)}", f"📋 Total: {len(res)}\n"]
        for _, ep in res.head(10).iterrows():
            linhas.append(f"  Ep. {ep['id']} — {ep['titulo']} ({ep['tipo']})")
        return "\n".join(linhas)

    def _busca_semantica(self, texto: str) -> str:
        t_limpo = self._pre_processar_para_busca(texto)
        v_pergunta = self._vetorizador.transform([t_limpo])
        sim = cosine_similarity(v_pergunta, self._matriz_tfidf).flatten()
        idx = int(sim.argmax())
        
        if sim[idx] < 0.15:
            return "Não encontrei nada específico. Tente citar um personagem ou luta!"

        ep = self._df.iloc[idx]
        return (f"📺 Episódio {ep['id']} — {ep['titulo']}\n"
                f"📁 {ep['saga']} > {ep['arco']} | Tipo: {ep['tipo']}\n"
                f"📝 {ep['resumo']}")