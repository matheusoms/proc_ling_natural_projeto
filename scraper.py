"""
scraper.py — Módulo de Web Scraping para o Chatbot One Piece

Responsável por extrair dados de episódios das sagas pré-time-skip
da Fandom Wiki de One Piece (PT-BR) e salvar em dataset_episodios.json.

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

URLS_SAGAS: list[tuple[str, str]] = [
    (
        "https://onepiece.fandom.com/pt/wiki/Guia_de_Epis%C3%B3dios/Saga_East_Blue",
        "Saga East Blue",
    ),
    (
        "https://onepiece.fandom.com/pt/wiki/Guia_de_Epis%C3%B3dios/Saga_Alabasta",
        "Saga Alabasta",
    ),
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


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------


def _criar_scraper() -> cloudscraper.CloudScraper:
    """
    Cria e retorna uma instância configurada do cloudscraper.

    O cloudscraper emula um browser real (Chrome) e resolve os desafios
    JavaScript do Cloudflare, contornando o erro 403 do Fandom Wiki.

    Retorno
    -------
    cloudscraper.CloudScraper
        Instância pronta para uso com fingerprint de Chrome/Windows.
    """
    return cloudscraper.create_scraper(
        browser={
            "browser": "chrome",
            "platform": "windows",
            "mobile": False,
        }
    )


def _extrair_cor_background(elemento: Tag) -> str | None:
    """
    Extrai a cor de fundo de um elemento HTML a partir de style ou bgcolor.

    Parâmetros
    ----------
    elemento : Tag
        Elemento BeautifulSoup (normalmente um <td>).

    Retorno
    -------
    str | None
        Cor hex minúscula (ex: '#f2f2f2') ou None.
    """
    style = elemento.get("style", "")
    if style:
        match = re.search(
            r'background(?:-color)?\s*:\s*(#[0-9a-fA-F]{3,8})',
            style,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).lower()

    bgcolor = elemento.get("bgcolor", "")
    if bgcolor and bgcolor.startswith("#"):
        return bgcolor.lower()

    return None


def _determinar_tipo(cor: str | None, resumo: str) -> str:
    """
    Determina o tipo do episódio pela cor de fundo e pelo conteúdo do resumo.

    A marcação '(Semi-Filler)' no resumo tem prioridade sobre a cor.

    Parâmetros
    ----------
    cor : str | None
        Cor hex do fundo da célula de número (ex: '#ccffff').
    resumo : str
        Texto completo do resumo do episódio.

    Retorno
    -------
    str
        Tipo: 'Canon', 'Filler', 'Semi-Filler', 'OVA', 'Filme', etc.
    """
    if "(Semi-Filler)" in resumo or "(semi-filler)" in resumo.lower():
        return "Semi-Filler"
    return MAPA_CORES_TIPO.get(cor, "Desconhecido") if cor else "Desconhecido"


def _extrair_titulo(th: Tag | None) -> str:
    """
    Extrai o título PT-BR do <th> de episódio, ignorando o título japonês.

    O título japonês sempre aparece após uma tag <br>. Iteramos os filhos
    e paramos ao encontrar o primeiro <br>.

    Parâmetros
    ----------
    th : Tag | None
        Elemento <th> com o título do episódio.

    Retorno
    -------
    str
        Título em português, sem o texto japonês.
    """
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
    """
    Verifica se um elemento <table> é uma tabela de episódio do Fandom.

    As tabelas de episódio possuem as classes 'collapsible' e 'collapsed'.

    Parâmetros
    ----------
    tabela : Tag
        Elemento <table>.

    Retorno
    -------
    bool
        True se for tabela de episódio.
    """
    classes = tabela.get("class", [])
    return "collapsible" in classes and "collapsed" in classes


def _processar_tabela_episodio(
    tabela: Tag,
    arco_atual: str,
    saga_atual: str,
) -> dict | None:
    """
    Extrai os dados de um episódio de uma tabela collapsible.

    Estrutura esperada:
      Linha 1: <td> número | <th> título (PT + JP separados por <br>) | <td> data
      Linha 2: <td colspan=3> resumo

    Parâmetros
    ----------
    tabela : Tag
        Elemento <table> do episódio.
    arco_atual : str
        Nome do arco vigente.
    saga_atual : str
        Nome da saga vigente.

    Retorno
    -------
    dict | None
        Dicionário do episódio ou None se a tabela não tiver o formato esperado.
    """
    # find_all("tr") sem recursive=False para traversar o <tbody> intermediário
    linhas = tabela.find_all("tr")

    if len(linhas) < 2:
        return None

    l1 = linhas[0]
    células = l1.find_all(["td", "th"])

    if len(células) < 3:
        return None

    td_numero = células[0]
    th_titulo = células[1]
    td_data   = células[2]

    numero = td_numero.get_text(strip=True)
    titulo = _extrair_titulo(th_titulo)
    data   = re.sub(r'\[\d+\]', '', td_data.get_text(strip=True)).strip()
    cor    = _extrair_cor_background(td_numero)

    l2 = linhas[1]
    td_resumo = l2.find("td")
    resumo = td_resumo.get_text(strip=True) if td_resumo else ""

    tipo = _determinar_tipo(cor, resumo)

    return {
        "id":     numero,
        "titulo": titulo,
        "data":   data,
        "resumo": resumo,
        "arco":   arco_atual,
        "saga":   saga_atual,
        "tipo":   tipo,
    }


# ---------------------------------------------------------------------------
# Funções públicas obrigatórias
# ---------------------------------------------------------------------------


def scrape_pagina(url: str, nome_saga: str, scraper: cloudscraper.CloudScraper) -> list[dict]:
    """
    Raspa uma única página de saga e retorna lista de episódios.

    ESTRATÉGIA DE PARSING — por que usamos .descendants:
    ----------------------------------------------------------
    O HTML do Fandom intercala <div> de anúncios entre os <h3> e as <table>
    correspondentes. O .find_all(["h2","h3","table"]) retorna uma lista plana
    onde não é possível garantir que o h3 detectado é o contexto correto de
    uma tabela específica.

    O .descendants percorre o DOM em profundidade na ordem exata do documento,
    preservando o rastreamento correto de saga/arco para cada tabela.

    O set `tabelas_vistas` evita processar a mesma tabela mais de uma vez,
    já que .descendants visita tanto o nó <table> quanto seus descendentes.

    Parâmetros
    ----------
    url : str
        URL completa da página da saga.
    nome_saga : str
        Nome da saga (valor inicial e identificador nos logs).
    scraper : cloudscraper.CloudScraper
        Instância compartilhada do cloudscraper.

    Retorno
    -------
    list[dict]
        Lista de episódios com: id, titulo, data, resumo, arco, saga, tipo.
    """
    episodios: list[dict] = []
    saga_atual = nome_saga
    arco_atual = nome_saga

    print(f"  → Raspando: {nome_saga} ...")

    try:
        resposta = scraper.get(url, timeout=30)
        resposta.raise_for_status()
    except Exception as erro:
        print(f"  [ERRO] Falha ao acessar '{url}': {erro}")
        return episodios

    soup = BeautifulSoup(resposta.content, "html.parser")
    conteudo = soup.find("div", class_="mw-parser-output")

    if not conteudo:
        print(f"  [AVISO] div.mw-parser-output não encontrada em '{url}'.")
        return episodios

    tabelas_vistas: set[int] = set()

    for elemento in conteudo.descendants:

        if not isinstance(elemento, Tag):
            continue

        if elemento.name == "h2":
            texto = re.sub(r'\[.*?\]', '', elemento.get_text()).strip()
            if texto and "Referências" not in texto and "Índice" not in texto:
                saga_atual = texto
                arco_atual = texto  # reset do arco ao mudar de saga

        elif elemento.name == "h3":
            texto = re.sub(r'\[.*?\]', '', elemento.get_text()).strip()
            if texto:
                arco_atual = texto

        elif elemento.name == "table" and _e_tabela_episodio(elemento):
            elem_id = id(elemento)
            if elem_id in tabelas_vistas:
                continue
            tabelas_vistas.add(elem_id)

            episodio = _processar_tabela_episodio(elemento, arco_atual, saga_atual)
            if episodio:
                episodios.append(episodio)

    print(f"  ✓ {len(episodios)} episódios encontrados em '{nome_saga}'.")
    return episodios


def scrape_todas_sagas() -> list[dict]:
    """
    Itera todas as URLs configuradas e retorna a lista completa de episódios.

    Reutiliza uma única instância do cloudscraper (mantém o contexto do
    handshake Cloudflare entre páginas). Aplica delay de 2s entre requisições.

    Retorno
    -------
    list[dict]
        Lista acumulada de todos os episódios de todas as sagas.
    """
    todos: list[dict] = []

    print("\n🏴‍☠️  Iniciando scraping das sagas de One Piece...\n")

    scraper = _criar_scraper()

    for indice, (url, nome_saga) in enumerate(URLS_SAGAS):
        episodios_saga = scrape_pagina(url, nome_saga, scraper)
        todos.extend(episodios_saga)

        if indice < len(URLS_SAGAS) - 1:
            print("  ⏳ Aguardando 2 segundos...\n")
            time.sleep(2)

    print(f"\n✅ Scraping concluído! Total: {len(todos)} episódios.\n")
    return todos


def salvar_dataset(episodios: list[dict], path: str = "dataset_episodios.json") -> None:
    """
    Salva a lista de episódios em JSON formatado (indentação de 2 espaços).

    Parâmetros
    ----------
    episodios : list[dict]
        Lista de dicionários a serializar.
    path : str, optional
        Caminho do arquivo de saída. Padrão: 'dataset_episodios.json'.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(episodios, f, ensure_ascii=False, indent=2)
        print(f"💾 Dataset salvo: '{path}' ({len(episodios)} episódios).")
    except OSError as erro:
        print(f"[ERRO] Não foi possível salvar '{path}': {erro}")


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    caminho = "dataset_episodios.json"

    if os.path.exists(caminho):
        resposta = input(
            f"\n⚠️  '{caminho}' já existe. Re-raspar e sobrescrever? [s/N]: "
        ).strip().lower()
        if resposta != "s":
            print("✅ Usando dataset existente.")
        else:
            salvar_dataset(scrape_todas_sagas(), caminho)
    else:
        salvar_dataset(scrape_todas_sagas(), caminho)