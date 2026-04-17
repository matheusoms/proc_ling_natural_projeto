"""
scraper.py — Módulo de Web Scraping para o Chatbot One Piece

Responsável por extrair dados de episódios das sagas pré-time-skip
da Fandom Wiki de One Piece (PT-BR) e salvar em dataset_episodios.json.

ESTRATÉGIA HÍBRIDA DE PARSING:
  - Saga East Blue e Saga Alabasta usam tabelas 'collapsible collapsed'
    com dados completos (id, título, data, resumo) inline na página.
    
  - A partir da Saga Ilha do Céu, os episódios são listados como <ul><li>
    com apenas o número, título e tipo textual. O resumo é obtido acessando
    a página individual de cada episódio (ex: /pt/wiki/Episódio_139).

Usa cloudscraper para contornar a proteção Cloudflare/anti-bot do Fandom.

Instalação:
    pip install cloudscraper beautifulsoup4

Autor: Projeto PLN — FATEC
"""

import json
import time
import re
import os

import cloudscraper
from bs4 import BeautifulSoup, Tag


# ---------------------------------------------------------------------------
# Constantes de configuração
# ---------------------------------------------------------------------------

BASE_URL = "https://onepiece.fandom.com"

# Sagas que usam o formato de TABELAS collapsible (formato antigo)
SAGAS_FORMATO_TABELA: list[tuple[str, str]] = [
    (
        "https://onepiece.fandom.com/pt/wiki/Guia_de_Epis%C3%B3dios/Saga_East_Blue",
        "Saga East Blue",
    ),
    (
        "https://onepiece.fandom.com/pt/wiki/Guia_de_Epis%C3%B3dios/Saga_Alabasta",
        "Saga Alabasta",
    ),
]

# Sagas que usam o formato de LISTA <ul><li> (formato novo — requer visita individual)
SAGAS_FORMATO_LISTA: list[tuple[str, str]] = [
    (
        "https://onepiece.fandom.com/pt/wiki/Guia_de_Epis%C3%B3dios/Saga_Ilha_do_C%C3%A9u",
        "Saga Ilha do Céu",
    ),
    (
        "https://onepiece.fandom.com/pt/wiki/Guia_de_Epis%C3%B3dios/Saga_Water_7",
        "Saga Water 7",
    ),
    (
        "https://onepiece.fandom.com/pt/wiki/Guia_de_Epis%C3%B3dios/Saga_Thriller_Bark",
        "Saga Thriller Bark",
    ),
    (
        "https://onepiece.fandom.com/pt/wiki/Guia_de_Epis%C3%B3dios/Saga_C%C3%BApula_da_Guerra",
        "Saga Cúpula da Guerra",
    ),
    (
        "https://onepiece.fandom.com/pt/wiki/Guia_de_Epis%C3%B3dios/Saga_Ilha_dos_Homens-Peixe",
        "Saga Ilha dos Homens-Peixe",
    ),
]

# Mapeamento de cor de fundo → tipo (usado apenas no formato tabela)
MAPA_CORES_TIPO: dict[str, str] = {
    "#f2f2f2": "Canon",
    "#ccffff": "Filler",
    "#80ff80": "Recap",
    "#ffff79": "Especial",
    "#abcdef": "Filme",
    "#ff9f80": "Especial TV",
    "#ffcccc": "OVA",
    "#eecfdd": "Especial Mobile",
    "#ffcc99": "Curta-Metragem",
}

# Delay entre requisições de páginas individuais de episódio (segundos)
DELAY_EPISODIO = 1.0
# Delay entre sagas completas (segundos)
DELAY_SAGA = 2.0


# ---------------------------------------------------------------------------
# Setup do scraper
# ---------------------------------------------------------------------------


def _criar_scraper() -> cloudscraper.CloudScraper:
    """
    Cria uma instância do cloudscraper simulando Chrome/Windows.

    Retorno
    -------
    cloudscraper.CloudScraper
    """
    return cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )


def _get_soup(url: str, scraper: cloudscraper.CloudScraper) -> BeautifulSoup | None:
    """
    Realiza uma requisição GET e retorna o BeautifulSoup da página.

    Parâmetros
    ----------
    url : str
        URL a ser acessada.
    scraper : cloudscraper.CloudScraper
        Instância compartilhada do scraper.

    Retorno
    -------
    BeautifulSoup | None
        Objeto soup ou None em caso de erro.
    """
    try:
        resposta = scraper.get(url, timeout=30)
        resposta.raise_for_status()
        return BeautifulSoup(resposta.content, "html.parser")
    except Exception as erro:
        print(f"    [ERRO] Falha ao acessar '{url}': {erro}")
        return None


# ---------------------------------------------------------------------------
# Helpers comuns
# ---------------------------------------------------------------------------


def _extrair_cor_background(elemento: Tag) -> str | None:
    """Extrai a cor de fundo de um elemento via style ou bgcolor."""
    style = elemento.get("style", "")
    if style:
        match = re.search(
            r'background(?:-color)?\s*:\s*(#[0-9a-fA-F]{3,8})',
            style, re.IGNORECASE,
        )
        if match:
            return match.group(1).lower()
    bgcolor = elemento.get("bgcolor", "")
    if bgcolor and bgcolor.startswith("#"):
        return bgcolor.lower()
    return None


def _determinar_tipo_por_cor(cor: str | None, resumo: str = "") -> str:
    """Determina o tipo pelo mapa de cores. Semi-Filler tem prioridade textual."""
    if "(Semi-Filler)" in resumo or "(semi-filler)" in resumo.lower():
        return "Semi-Filler"
    return MAPA_CORES_TIPO.get(cor, "Desconhecido") if cor else "Desconhecido"


def _determinar_tipo_por_texto(texto_li: str) -> str:
    """
    Determina o tipo de episódio a partir do texto do <li> no formato lista.

    O texto pode conter marcações como '(Filler)', '(Semi-Filler)', etc.
    Episódios sem marcação são Canon.

    Parâmetros
    ----------
    texto_li : str
        Texto completo do elemento <li>.

    Retorno
    -------
    str
        Tipo do episódio.
    """
    texto_lower = texto_li.lower()
    if "(semi-filler)" in texto_lower:
        return "Semi-Filler"
    if "(filler)" in texto_lower:
        return "Filler"
    if "(recap)" in texto_lower or "recapitulação" in texto_lower:
        return "Recap"
    if "(especial tv)" in texto_lower:
        return "Especial TV"
    if "(ova)" in texto_lower:
        return "OVA"
    if "(filme)" in texto_lower:
        return "Filme"
    if "(especial)" in texto_lower:
        return "Especial"
    return "Canon"


def _extrair_titulo_th(th: Tag | None) -> str:
    """Extrai título PT-BR de um <th>, ignorando texto japonês após <br>."""
    if th is None:
        return ""
    partes: list[str] = []
    for filho in th.children:
        if isinstance(filho, Tag) and filho.name == "br":
            break
        texto = filho.get_text(strip=True) if isinstance(filho, Tag) else str(filho).strip()
        if texto:
            partes.append(texto)
    return " ".join(partes).strip()


def _e_tabela_episodio(tabela: Tag) -> bool:
    """Verifica se a tabela é uma tabela de episódio (collapsible collapsed)."""
    classes = tabela.get("class", [])
    return "collapsible" in classes and "collapsed" in classes


# ---------------------------------------------------------------------------
# FORMATO 1: Tabelas collapsible (Saga East Blue, Saga Alabasta)
# ---------------------------------------------------------------------------


def _processar_tabela_episodio(
    tabela: Tag, arco: str, saga: str
) -> dict | None:
    """
    Extrai dados de episódio de uma tabela collapsible do formato antigo.

    Parâmetros
    ----------
    tabela : Tag
    arco : str
    saga : str

    Retorno
    -------
    dict | None
    """
    linhas = tabela.find_all("tr")
    if len(linhas) < 2:
        return None

    células = linhas[0].find_all(["td", "th"])
    if len(células) < 3:
        return None

    td_numero, th_titulo, td_data = células[0], células[1], células[2]

    numero = td_numero.get_text(strip=True)
    titulo = _extrair_titulo_th(th_titulo)
    data   = re.sub(r'\[\d+\]', '', td_data.get_text(strip=True)).strip()
    cor    = _extrair_cor_background(td_numero)

    td_resumo = linhas[1].find("td")
    resumo = td_resumo.get_text(strip=True) if td_resumo else ""

    tipo = _determinar_tipo_por_cor(cor, resumo)

    return {
        "id":     numero,
        "titulo": titulo,
        "data":   data,
        "resumo": resumo,
        "arco":   arco,
        "saga":   saga,
        "tipo":   tipo,
    }


def scrape_pagina_formato_tabela(
    url: str, nome_saga: str, scraper: cloudscraper.CloudScraper
) -> list[dict]:
    """
    Raspa sagas no formato de tabelas collapsible (East Blue, Alabasta).

    Itera os descendentes do conteúdo em ordem de documento, rastreando
    h2/h3 para manter o contexto de saga/arco correto.

    Parâmetros
    ----------
    url : str
    nome_saga : str
    scraper : cloudscraper.CloudScraper

    Retorno
    -------
    list[dict]
    """
    episodios: list[dict] = []
    saga_atual = nome_saga
    arco_atual = nome_saga

    print(f"  → [TABELA] Raspando: {nome_saga} ...")

    soup = _get_soup(url, scraper)
    if not soup:
        return episodios

    conteudo = soup.find("div", class_="mw-parser-output")
    if not conteudo:
        print(f"  [AVISO] mw-parser-output não encontrado em '{url}'.")
        return episodios

    tabelas_vistas: set[int] = set()

    for elemento in conteudo.descendants:
        if not isinstance(elemento, Tag):
            continue

        if elemento.name == "h2":
            texto = re.sub(r'\[.*?\]', '', elemento.get_text()).strip()
            if texto and "Referências" not in texto and "Índice" not in texto:
                saga_atual = texto
                arco_atual = texto

        elif elemento.name == "h3":
            texto = re.sub(r'\[.*?\]', '', elemento.get_text()).strip()
            if texto:
                arco_atual = texto

        elif elemento.name == "table" and _e_tabela_episodio(elemento):
            eid = id(elemento)
            if eid in tabelas_vistas:
                continue
            tabelas_vistas.add(eid)

            ep = _processar_tabela_episodio(elemento, arco_atual, saga_atual)
            if ep:
                episodios.append(ep)

    print(f"  ✓ {len(episodios)} episódios em '{nome_saga}'.")
    return episodios


# ---------------------------------------------------------------------------
# FORMATO 2: Lista <ul><li> com visita individual (Ilha do Céu em diante)
# ---------------------------------------------------------------------------


def _extrair_resumo_breve_pagina_episodio(
    soup: BeautifulSoup,
) -> tuple[str, str]:
    """
    Extrai o resumo breve e a data de exibição da página individual do episódio.

    Na página do episódio (ex: /pt/wiki/Episódio_139), o resumo breve está
    no primeiro <p> de conteúdo após o infobox. A data está no infobox
    dentro do elemento com data-source="Airdate".

    Parâmetros
    ----------
    soup : BeautifulSoup
        Soup da página individual do episódio.

    Retorno
    -------
    tuple[str, str]
        (resumo_breve, data_exibicao) — strings vazias se não encontrado.
    """
    resumo = ""
    data   = ""

    conteudo = soup.find("div", class_="mw-parser-output")
    if not conteudo:
        return resumo, data

    # --- Data: campo Airdate no infobox ---
    airdate_div = conteudo.find("div", {"data-source": "Airdate"})
    if airdate_div:
        valor = airdate_div.find(class_="pi-data-value")
        if valor:
            data = valor.get_text(strip=True)

    # --- Resumo: primeiro <p> com conteúdo real após o infobox ---
    # Pular parágrafos vazios, que são apenas o infobox ou espaçadores
    for p in conteudo.find_all("p"):
        texto = p.get_text(strip=True)
        # Ignorar parágrafos curtos/vazios e o parágrafo de título do episódio
        if texto and len(texto) > 60 and "Predefinição:" not in texto:
            resumo = texto
            break

    # Fallback: tentar a seção "Resumo Breve" se existir
    if not resumo:
        h2_resumo = conteudo.find("span", {"id": "Resumo_Breve"})
        if h2_resumo:
            proximo = h2_resumo.find_next("p")
            if proximo:
                resumo = proximo.get_text(strip=True)

    return resumo, data


def _extrair_entradas_lista(
    conteudo: Tag,
) -> list[tuple[str, str, str, str, str]]:
    """
    Extrai as entradas de episódio de uma página no formato <ul><li>.

    Cada <li> tem o formato:
      "Episódio 139: Título do episódio (Tipo)"
    com um link <a> apontando para /pt/wiki/Episódio_NNN.

    Parâmetros
    ----------
    conteudo : Tag
        Div mw-parser-output da página da saga.

    Retorno
    -------
    list[tuple[str, str, str, str, str]]
        Lista de (numero_ep, titulo, url_episodio, tipo, arco_atual).
    """
    entradas: list[tuple[str, str, str, str, str]] = []
    arco_atual = ""

    for elemento in conteudo.descendants:
        if not isinstance(elemento, Tag):
            continue

        if elemento.name == "h3":
            texto = re.sub(r'\[.*?\]', '', elemento.get_text()).strip()
            if texto:
                arco_atual = texto

        elif elemento.name == "li":
            # Evitar processar <li> filhos de listas de navegação/TOC
            pai = elemento.parent
            if isinstance(pai, Tag) and pai.get("id") in ("toc", "sticky-toc"):
                continue

            link = elemento.find("a", href=re.compile(r'/pt/wiki/Epis%C3%B3dio_\d+'))
            if not link:
                continue

            texto_completo = elemento.get_text(strip=True)
            href = link.get("href", "")

            # Extrair número do episódio da URL ou do texto
            match_num = re.search(r'Epis%C3%B3dio_(\d+)', href)
            numero = match_num.group(1) if match_num else ""

            # Extrair título: texto após "Episódio NNN: "
            match_titulo = re.search(r'Episódio\s+\d+\s*:\s*(.+)', texto_completo)
            if match_titulo:
                titulo_raw = match_titulo.group(1)
                # Remover marcação de tipo entre parênteses no final
                titulo = re.sub(r'\s*\((Filler|Semi-Filler|Recap|Especial.*?|OVA|Filme)\)\s*$', '', titulo_raw, flags=re.IGNORECASE).strip()
            else:
                titulo = link.get_text(strip=True)

            tipo = _determinar_tipo_por_texto(texto_completo)
            url_ep = BASE_URL + href

            entradas.append((numero, titulo, url_ep, tipo, arco_atual))

    return entradas


def scrape_pagina_formato_lista(
    url: str,
    nome_saga: str,
    scraper: cloudscraper.CloudScraper,
) -> list[dict]:
    """
    Raspa sagas no formato de lista <ul><li> com visita a cada episódio.

    Para cada entrada na lista, acessa a página individual do episódio
    para obter o resumo breve e a data de exibição.

    Parâmetros
    ----------
    url : str
        URL da página da saga.
    nome_saga : str
    scraper : cloudscraper.CloudScraper

    Retorno
    -------
    list[dict]
    """
    episodios: list[dict] = []

    print(f"  → [LISTA] Raspando índice: {nome_saga} ...")

    soup_index = _get_soup(url, scraper)
    if not soup_index:
        return episodios

    conteudo = soup_index.find("div", class_="mw-parser-output")
    if not conteudo:
        print(f"  [AVISO] mw-parser-output não encontrado em '{url}'.")
        return episodios

    entradas = _extrair_entradas_lista(conteudo)
    total = len(entradas)
    print(f"  ✓ {total} episódios encontrados no índice. Iniciando visitas individuais...")

    for i, (numero, titulo, url_ep, tipo, arco) in enumerate(entradas, start=1):
        print(f"    [{i:03d}/{total}] Ep.{numero} — {titulo[:55]}...")

        soup_ep = _get_soup(url_ep, scraper)
        if soup_ep:
            resumo, data = _extrair_resumo_breve_pagina_episodio(soup_ep)
        else:
            resumo, data = "", ""

        episodios.append({
            "id":     numero,
            "titulo": titulo,
            "data":   data,
            "resumo": resumo,
            "arco":   arco,
            "saga":   nome_saga,
            "tipo":   tipo,
        })

        # Delay respeitoso entre requisições de episódios individuais
        if i < total:
            time.sleep(DELAY_EPISODIO)

    print(f"  ✓ {len(episodios)} episódios completos de '{nome_saga}'.")
    return episodios


# ---------------------------------------------------------------------------
# Funções públicas obrigatórias
# ---------------------------------------------------------------------------


def scrape_todas_sagas() -> list[dict]:
    """
    Itera todas as sagas configuradas usando a estratégia correta para cada uma.

    Sagas East Blue e Alabasta: formato tabela (dados inline, rápido).
    Sagas a partir de Ilha do Céu: formato lista (visita individual, mais lento).

    Retorno
    -------
    list[dict]
        Lista acumulada de todos os episódios de todas as sagas.
    """
    todos: list[dict] = []

    print("\n Iniciando scraping das sagas de One Piece...\n")
    scraper = _criar_scraper()

    # --- Formato Tabela ---
    for indice, (url, nome_saga) in enumerate(SAGAS_FORMATO_TABELA):
        eps = scrape_pagina_formato_tabela(url, nome_saga, scraper)
        todos.extend(eps)
        print(f" Aguardando {DELAY_SAGA}s...\n")
        time.sleep(DELAY_SAGA)

    # --- Formato Lista ---
    for indice, (url, nome_saga) in enumerate(SAGAS_FORMATO_LISTA):
        eps = scrape_pagina_formato_lista(url, nome_saga, scraper)
        todos.extend(eps)
        if indice < len(SAGAS_FORMATO_LISTA) - 1:
            print(f"Aguardando {DELAY_SAGA}s antes da próxima saga...\n")
            time.sleep(DELAY_SAGA)

    print(f"\n Scraping concluído! Total: {len(todos)} episódios.\n")
    return todos


def salvar_dataset(episodios: list[dict], path: str = "dataset_episodios.json") -> None:
    """
    Salva a lista de episódios em JSON formatado (indentação de 2 espaços).

    Parâmetros
    ----------
    episodios : list[dict]
    path : str, optional
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(episodios, f, ensure_ascii=False, indent=2)
        print(f"Dataset salvo: '{path}' ({len(episodios)} episódios).")
    except OSError as erro:
        print(f"[ERRO] Não foi possível salvar '{path}': {erro}")


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    caminho = "dataset_episodios.json"

    if os.path.exists(caminho):
        resposta = input(
            f"\n '{caminho}' já existe. Re-raspar e sobrescrever? [s/N]: "
        ).strip().lower()
        if resposta != "s":
            print("Usando dataset existente.")
        else:
            salvar_dataset(scrape_todas_sagas(), caminho)
    else:
        salvar_dataset(scrape_todas_sagas(), caminho)