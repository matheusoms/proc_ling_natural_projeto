"""
debug_parser.py — Script de diagnóstico do parser BeautifulSoup

Cole o HTML da página no arquivo 'east_blue_test.html' e execute este script
para verificar o que o BeautifulSoup está enxergando na estrutura da página.

Uso:
    python debug_parser.py
"""

from bs4 import BeautifulSoup
import re

# HTML mínimo da Saga East Blue para teste local (cole o conteúdo do site aqui)
# Para testar sem fazer requisição, salve o HTML em 'east_blue_test.html'
HTML_TESTE = """
<div class="mw-content-ltr mw-parser-output" lang="pt-BR" dir="ltr">
<h2><span class="mw-headline" id="Saga_East_Blue">Saga East Blue</span></h2>
<h3><span class="mw-headline" id="OVA_de_One_Piece">OVA de One Piece</span></h3>
<table class="wikitable" cellpadding="2" border="1"></table>
<table class="collapsible collapsed" width="100%" style="font-size: 90%;" border="1" cellpadding="3">
<tbody><tr>
<td width="5%" style="background:#FFCCCC; color:#000000" align="center">1</td>
<th width="83%" align="left" style="background:#FFCCCC; color:#000000">One Piece - Derrote-o! O Pirata Ganzack<br>One Piece Taose!</th>
<td width="12%" style="background:#FFCCCC; color:#000000">26 de Julho de 1998</td>
</tr>
<tr>
<td colspan="3" style="color:#000000">Enquanto Luffy e sua tripulação estão famintos...</td>
</tr></tbody></table>
<h3><span class="mw-headline" id="Arco_Romance_Dawn">Arco Romance Dawn</span></h3>
<table class="wikitable" cellpadding="2" border="1"></table>
<table class="collapsible collapsed" style="width:100%; font-size:90%;" border="1" cellpadding="2">
<tbody><tr>
<td style="width:5%; text-align:center; background:#F2F2F2; color:#000000">001</td>
<th style="width:83%; text-align:left; background:#F2F2F2; color:#000000">Eu Sou Luffy! O Homem que Vai Ser O Rei dos Piratas!<br>Ore wa Rufi!</th>
<td style="width:12%; background:#F2F2F2; color:#000000">20 de Outubro, de 1999</td>
</tr>
<tr>
<td colspan="3" style="color:#000000">Um bando pirata liderado por Alvida...</td>
</tr></tbody></table>
</div>
"""


def diagnosticar_html(html: str) -> None:
    """
    Analisa a estrutura do HTML e imprime diagnósticos detalhados.
    """
    soup = BeautifulSoup(html, "html.parser")
    conteudo = soup.find("div", class_="mw-parser-output")

    if not conteudo:
        print("❌ ERRO: div.mw-parser-output não encontrada!")
        return

    print("✅ div.mw-parser-output encontrada.\n")

    # --- Diagnóstico 1: Contar elementos-chave ---
    h2s = conteudo.find_all("h2")
    h3s = conteudo.find_all("h3")
    tabelas_collapsible = conteudo.find_all(
        "table", class_=lambda c: c and "collapsible" in c and "collapsed" in c
    )
    tabelas_wikitable = conteudo.find_all("table", class_="wikitable")

    print(f"📊 Diagnóstico de elementos encontrados:")
    print(f"   H2 (sagas):              {len(h2s)}")
    print(f"   H3 (arcos):              {len(h3s)}")
    print(f"   Tabelas collapsible:     {len(tabelas_collapsible)}")
    print(f"   Tabelas wikitable:       {len(tabelas_wikitable)}")
    print()

    # --- Diagnóstico 2: Testar detecção de classes ---
    print("🔍 Testando detecção de classes nas tabelas collapsible:")
    for i, tabela in enumerate(tabelas_collapsible[:3]):
        classes = tabela.get("class", [])
        print(f"   Tabela {i+1}: classes = {classes}")
        linhas = tabela.find_all("tr", recursive=False)
        print(f"            linhas diretas (recursive=False): {len(linhas)}")
        linhas_todas = tabela.find_all("tr")
        print(f"            linhas totais (recursive=True):  {len(linhas_todas)}")

        if linhas_todas:
            l1 = linhas_todas[0]
            células = l1.find_all(["td", "th"], recursive=False)
            print(f"            células na linha 1: {len(células)}")
            if células:
                td_num = células[0]
                print(f"            style da td número: '{td_num.get('style', '')}'")
                print(f"            bgcolor da td número: '{td_num.get('bgcolor', '')}'")
    print()

    # --- Diagnóstico 3: Testar iteração em ordem do documento ---
    print("📋 Iteração em ordem do documento (children do conteudo):")
    arco_atual = "N/A"
    saga_atual = "N/A"
    contagem = 0

    # MÉTODO CORRETO: usar .children ou iterar via .find_all com ordenação
    for elemento in conteudo.descendants:
        if not hasattr(elemento, 'name') or elemento.name is None:
            continue

        if elemento.name == "h2":
            texto = re.sub(r'\[.*?\]', '', elemento.get_text()).strip()
            if texto and "Referências" not in texto:
                saga_atual = texto
                print(f"   [H2] Saga: {saga_atual}")

        elif elemento.name == "h3":
            texto = re.sub(r'\[.*?\]', '', elemento.get_text()).strip()
            if texto:
                arco_atual = texto
                print(f"      [H3] Arco: {arco_atual}")

        elif elemento.name == "table":
            classes = elemento.get("class", [])
            if "collapsible" in classes and "collapsed" in classes:
                # Verificar se este elemento é filho DIRETO (não aninhado)
                if elemento.parent == conteudo or elemento.parent.name in ["div", "body"]:
                    contagem += 1
                    linhas = elemento.find_all("tr")
                    if linhas:
                        td = linhas[0].find("td")
                        num = td.get_text(strip=True) if td else "?"
                        print(f"         [TABELA ep.{num}] saga={saga_atual} | arco={arco_atual}")

    print(f"\n✅ Total de episódios detectados: {contagem}")


if __name__ == "__main__":
    try:
        with open("east_blue_test.html", "r", encoding="utf-8") as f:
            html = f.read()
        print("📄 Usando arquivo 'east_blue_test.html'\n")
    except FileNotFoundError:
        print("📄 Arquivo não encontrado. Usando HTML de teste interno.\n")
        html = HTML_TESTE

    diagnosticar_html(html)