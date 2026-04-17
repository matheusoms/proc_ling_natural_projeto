"""
nlp_engine.py — Motor de NLP e Geração de Respostas para o Chatbot One Piece

Carrega o dataset gerado pelo scraper.py, processa perguntas em linguagem
natural e retorna respostas inteligentes usando filtros lógicos ou
similaridade semântica (TF-IDF + cosine similarity).

Autor: Projeto PLN — FATEC
"""

import json
import re
import unicodedata

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Funções auxiliares de normalização de texto
# ---------------------------------------------------------------------------


def _normalizar_texto(texto: str) -> str:
    """
    Normaliza um texto: converte para minúsculas e remove acentuação.

    Parâmetros
    ----------
    texto : str
        Texto original a ser normalizado.

    Retorno
    -------
    str
        Texto em minúsculas sem caracteres acentuados.
    """
    texto_lower = texto.lower()
    # Decompor unicode e remover diacríticos (acentos)
    normalizado = unicodedata.normalize("NFD", texto_lower)
    sem_acento = "".join(c for c in normalizado if unicodedata.category(c) != "Mn")
    return sem_acento


# ---------------------------------------------------------------------------
# Classe principal do chatbot
# ---------------------------------------------------------------------------


class OnePieceChatbot:
    """
    Chatbot especialista em One Piece com suporte a busca por filtros lógicos
    e busca semântica via TF-IDF + similaridade de cossenos.

    Todo o estado está encapsulado nesta classe — sem variáveis globais.
    """

    # Palavras-chave que indicam intenção de filtro estruturado
    KEYWORDS_FILTRO: list[str] = [
        "filler",
        "semi-filler",
        "canon",
        "recap",
        "quantos",
        "listar",
        "quais",
        "arco",
        "saga",
        "episodios de",
        "mostre",
        "ova",
        "filme",
        "especial",
        "curta",
    ]

    # Palavras-chave que indicam um tipo específico de episódio
    KEYWORDS_TIPO: list[str] = [
        "filler",
        "semi-filler",
        "canon",
        "recap",
        "ova",
        "filme",
        "especial",
        "curta-metragem",
        "curta",
        "especial tv",
        "especial mobile",
    ]

    # Mapeamento de palavra-chave normalizada → nome de tipo canônico no dataset
    MAPA_TIPO_CANONICO: dict[str, str] = {
        "filler": "Filler",
        "semi-filler": "Semi-Filler",
        "semifiller": "Semi-Filler",
        "canon": "Canon",
        "recap": "Recap",
        "ova": "OVA",
        "filme": "Filme",
        "especial": "Especial",
        "especial tv": "Especial TV",
        "especial mobile": "Especial Mobile",
        "curta-metragem": "Curta-Metragem",
        "curta": "Curta-Metragem",
    }

    def __init__(self, path: str = "dataset_episodios.json") -> None:
        """
        Inicializa o chatbot carregando dados, modelo NLP e vetorizador TF-IDF.

        Parâmetros
        ----------
        path : str, optional
            Caminho para o arquivo JSON com os dados dos episódios.
            Padrão: 'dataset_episodios.json'.

        Levanta
        -------
        FileNotFoundError
            Se o arquivo JSON não for encontrado.
        ValueError
            Se o dataset estiver vazio ou sem a coluna 'resumo'.
        """
        # --- Carregar dataset ---
        try:
            self._df: pd.DataFrame = pd.read_json(path, encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset não encontrado em '{path}'. "
                "Execute 'python scraper.py' primeiro para gerar o dataset."
            )
        except ValueError as erro:
            raise ValueError(f"Erro ao ler o dataset JSON: {erro}")

        if self._df.empty:
            raise ValueError("O dataset está vazio. Verifique o arquivo JSON.")

        # Garantir que coluna 'resumo' existe
        if "resumo" not in self._df.columns:
            raise ValueError("O dataset não contém a coluna 'resumo'.")

        # Preencher valores ausentes na coluna resumo
        self._df["resumo"] = self._df["resumo"].fillna("")

        # --- Carregar modelo spaCy ---
        try:
            self._nlp = spacy.load("pt_core_news_lg")
        except OSError:
            raise OSError(
                "Modelo spaCy 'pt_core_news_lg' não encontrado. "
                "Execute: python -m spacy download pt_core_news_lg"
            )

        # --- Inicializar e ajustar TF-IDF sobre os resumos ---
        self._vetorizador = TfidfVectorizer(
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2),
            strip_accents="unicode",
            lowercase=True,
        )
        self._matriz_tfidf = self._vetorizador.fit_transform(self._df["resumo"])

        # --- Construir índice de arcos e sagas para matching fuzzy ---
        self._arcos_unicos: list[str] = self._df["arco"].dropna().unique().tolist()
        self._sagas_unicas: list[str] = self._df["saga"].dropna().unique().tolist()

        # Versões normalizadas para comparação sem acento
        self._arcos_normalizados: list[str] = [_normalizar_texto(a) for a in self._arcos_unicos]
        self._sagas_normalizadas: list[str] = [_normalizar_texto(s) for s in self._sagas_unicas]

    # -----------------------------------------------------------------------
    # Métodos públicos principais
    # -----------------------------------------------------------------------

    def processar_pergunta(self, texto: str) -> dict:
        """
        Analisa a pergunta do usuário e extrai intenção e entidades.

        Parâmetros
        ----------
        texto : str
            Pergunta em linguagem natural do usuário.

        Retorno
        -------
        dict
            Dicionário com chaves:
            - 'intencao': 'filtro' ou 'semantica'
            - 'entidades': dict com 'arco', 'saga', 'tipo' (cada um str | None)
            - 'texto_normalizado': str em minúsculas sem acentos
        """
        texto_norm = _normalizar_texto(texto)

        # Detectar intenção: filtro vs semantica
        intencao = self._detectar_intencao(texto_norm)

        # Extrair entidades (arco, saga, tipo)
        entidades = self._extrair_entidades(texto, texto_norm)

        return {
            "intencao": intencao,
            "entidades": entidades,
            "texto_normalizado": texto_norm,
        }

    def gerar_resposta(self, texto_usuario: str) -> str:
        """
        Gera uma resposta para a pergunta do usuário.

        Decide entre busca por filtros lógicos (quando identificada intenção
        de filtro) ou busca semântica (caso contrário).

        Parâmetros
        ----------
        texto_usuario : str
            Texto da mensagem enviada pelo usuário.

        Retorno
        -------
        str
            Resposta formatada do chatbot.
        """
        analise = self.processar_pergunta(texto_usuario)
        intencao = analise["intencao"]
        entidades = analise["entidades"]

        if intencao == "filtro":
            return self._busca_por_filtros(entidades)
        else:
            return self._busca_semantica(texto_usuario)

    def listar_arcos(self) -> str:
        """
        Retorna uma string formatada com todos os arcos disponíveis no dataset,
        agrupados por saga — para orientar o usuário sobre o que pode consultar.

        Retorno
        -------
        str
            Texto formatado com a listagem de sagas e arcos.
        """
        linhas: list[str] = ["📚 **Arcos disponíveis no dataset:**\n"]

        for saga in self._sagas_unicas:
            arcos_da_saga = (
                self._df[self._df["saga"] == saga]["arco"]
                .dropna()
                .unique()
                .tolist()
            )
            linhas.append(f"🗺️  **{saga}**")
            for arco in arcos_da_saga:
                total = len(self._df[self._df["arco"] == arco])
                linhas.append(f"    • {arco} ({total} episódios)")
            linhas.append("")  # linha em branco entre sagas

        return "\n".join(linhas)

    # -----------------------------------------------------------------------
    # Métodos privados de análise
    # -----------------------------------------------------------------------

    def _detectar_intencao(self, texto_normalizado: str) -> str:
        """
        Detecta a intenção do usuário como 'filtro' ou 'semantica'.

        Parâmetros
        ----------
        texto_normalizado : str
            Texto já normalizado (sem acentos, em minúsculas).

        Retorno
        -------
        str
            'filtro' se qualquer keyword de filtro for encontrada,
            'semantica' caso contrário.
        """
        for kw in self.KEYWORDS_FILTRO:
            kw_norm = _normalizar_texto(kw)
            if kw_norm in texto_normalizado:
                return "filtro"
        return "semantica"

    def _extrair_entidades(self, texto_original: str, texto_normalizado: str) -> dict:
        """
        Extrai entidades relevantes da pergunta: arco, saga e tipo de episódio.

        Combina NER do spaCy com matching fuzzy baseado na lista de arcos/sagas
        do dataset.

        Parâmetros
        ----------
        texto_original : str
            Texto original da pergunta do usuário.
        texto_normalizado : str
            Versão normalizada do texto (sem acentos, minúsculas).

        Retorno
        -------
        dict
            Dicionário com chaves 'arco', 'saga', 'tipo' (str | None).
        """
        arco_encontrado: str | None = None
        saga_encontrada: str | None = None
        tipo_encontrado: str | None = None

        # --- Matching fuzzy: arcos ---
        for idx, arco_norm in enumerate(self._arcos_normalizados):
            # Remover prefixo "arco " para comparação mais flexível
            arco_sem_prefixo = arco_norm.replace("arco ", "")
            if arco_sem_prefixo in texto_normalizado or arco_norm in texto_normalizado:
                arco_encontrado = self._arcos_unicos[idx]
                break

        # --- Matching fuzzy: sagas ---
        for idx, saga_norm in enumerate(self._sagas_normalizadas):
            saga_sem_prefixo = saga_norm.replace("saga ", "")
            if saga_sem_prefixo in texto_normalizado or saga_norm in texto_normalizado:
                saga_encontrada = self._sagas_unicas[idx]
                break

        # --- Matching de tipo de episódio ---
        for kw in self.KEYWORDS_TIPO:
            kw_norm = _normalizar_texto(kw)
            if kw_norm in texto_normalizado:
                tipo_encontrado = self.MAPA_TIPO_CANONICO.get(kw_norm, kw.capitalize())
                break

        return {
            "arco": arco_encontrado,
            "saga": saga_encontrada,
            "tipo": tipo_encontrado,
        }

    # -----------------------------------------------------------------------
    # Métodos privados de busca
    # -----------------------------------------------------------------------

    def _busca_por_filtros(self, entidades: dict) -> str:
        """
        Filtra o DataFrame com base nas entidades identificadas e formata
        a resposta.

        Parâmetros
        ----------
        entidades : dict
            Dicionário com 'arco', 'saga' e 'tipo'.

        Retorno
        -------
        str
            Resposta formatada listando episódios encontrados ou mensagem
            de erro se nenhuma entidade reconhecida.
        """
        mascara = pd.Series([True] * len(self._df), index=self._df.index)
        filtros_aplicados: list[str] = []

        arco = entidades.get("arco")
        saga = entidades.get("saga")
        tipo = entidades.get("tipo")

        if arco:
            mascara &= self._df["arco"] == arco
            filtros_aplicados.append(f"arco='{arco}'")

        if saga and not arco:
            # Não aplicar filtro de saga se arco já foi aplicado (evitar conflito)
            mascara &= self._df["saga"] == saga
            filtros_aplicados.append(f"saga='{saga}'")

        if tipo:
            mascara &= self._df["tipo"].str.lower() == tipo.lower()
            filtros_aplicados.append(f"tipo='{tipo}'")

        resultados = self._df[mascara]

        # Nenhuma entidade reconhecida → estatísticas gerais
        if not filtros_aplicados:
            total_por_tipo = self._df["tipo"].value_counts()
            linhas = ["**Estatísticas gerais do dataset:**\n"]
            for tipo_ep, qtd in total_por_tipo.items():
                linhas.append(f"  • {tipo_ep}: {qtd} episódio(s)")
            linhas.append(f"\n **Total: {len(self._df)} episódios** em {len(self._sagas_unicas)} sagas.")
            linhas.append("\nUse 'Ver Arcos' para saber o que posso consultar!")
            return "\n".join(linhas)

        if resultados.empty:
            return (
                f"🔍 Nenhum episódio encontrado para: {', '.join(filtros_aplicados)}.\n"
                "Verifique o nome do arco/saga ou tente outros filtros."
            )

        total = len(resultados)
        exibir_ate = min(10, total)

        linhas = [
            f"🔎 Filtro: {' | '.join(filtros_aplicados)}",
            f"📋 Encontrados: **{total} episódios** (exibindo os {exibir_ate} primeiros)\n",
        ]

        for _, ep in resultados.head(10).iterrows():
            linhas.append(f"  Ep. {ep['id']} — {ep['titulo']} ({ep['tipo']})")

        if total > 10:
            linhas.append(f"\n  ... e mais {total - 10} episódio(s). Refine sua busca para ver menos resultados.")

        return "\n".join(linhas)

    def _busca_semantica(self, texto_usuario: str) -> str:
        """
        Realiza busca semântica usando TF-IDF e similaridade de cossenos.

        Parâmetros
        ----------
        texto_usuario : str
            Texto da pergunta do usuário.

        Retorno
        -------
        str
            Resposta formatada com o episódio mais similar, ou mensagem
            de baixa confiança quando o score for menor que 0.15.
        """
        LIMIAR_CONFIANCA = 0.15

        # Vetorizar a pergunta com o mesmo vetorizador já ajustado
        vetor_pergunta = self._vetorizador.transform([texto_usuario])

        # Calcular similaridade de cossenos com toda a matriz
        similaridades = cosine_similarity(vetor_pergunta, self._matriz_tfidf).flatten()

        # Encontrar o índice do episódio mais similar
        idx_melhor = int(similaridades.argmax())
        score = float(similaridades[idx_melhor])

        if score < LIMIAR_CONFIANCA:
            return (
                "Não encontrei episódios relacionados à sua pergunta. "
                "Tente ser mais específico!\n\n"
                "Dica: mencione personagens, eventos ou locais específicos de One Piece."
            )

        ep = self._df.iloc[idx_melhor]

        resposta = (
            f"📺 Episódio {ep['id']} — {ep['titulo']}\n"
            f"📁 {ep['saga']} > {ep['arco']} | Tipo: {ep['tipo']}\n"
            f"📅 {ep['data']}\n"
            f"📝 {ep['resumo']}"
        )
        return resposta
