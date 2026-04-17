# 🏴‍☠️ Chatbot Especialista de One Piece

> Um chatbot em Python que responde perguntas sobre episódios das sagas pré-time-skip de One Piece, usando Web Scraping, NLP com spaCy e busca semântica com TF-IDF.

---

## 📖 Descrição do Projeto

O **Chatbot Especialista de One Piece** é um sistema de perguntas e respostas construído com três módulos independentes:

| Módulo | Responsabilidade |
|---|---|
| `scraper.py` | Extrai dados de episódios da Fandom Wiki (PT-BR) e salva em JSON |
| `nlp_engine.py` | Processa perguntas em linguagem natural e gera respostas inteligentes |
| `main.py` | Interface gráfica Tkinter com tema escuro e carregamento assíncrono |

O chatbot suporta dois modos de busca:
- **Filtros lógicos** — Lista episódios por tipo (Canon, Filler, OVA...), arco ou saga
- **Busca semântica** — Encontra episódios por similaridade de conteúdo (TF-IDF + cosseno)

---

## ✅ Pré-requisitos

- **Python 3.12+** instalado e disponível no PATH
- Acesso à internet (necessário para o scraping e para baixar o modelo spaCy)
- Windows, Linux ou macOS

---

## 🚀 Instalação Passo a Passo

### 1. Clone ou baixe o repositório

```bash
# Via Git
git clone <url-do-repositorio>
cd proc_ling_natural_projeto

# Ou simplesmente extraia o ZIP na pasta desejada
```

### 2. (Recomendado) Crie um ambiente virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Instale as dependências Python

```bash
pip install -r requirements.txt
```

### 4. Baixe o modelo de NLP do spaCy

```bash
python -m spacy download pt_core_news_lg
```

> ⚠️ Este modelo tem ~560 MB. Certifique-se de ter espaço em disco e conexão estável.

---

## 🕷️ Executando o Scraper (Primeira Vez)

O scraper coleta dados das **7 sagas pré-time-skip** da Fandom Wiki e salva em `dataset_episodios.json`:

```bash
python scraper.py
```

**O que acontece:**
1. O scraper verifica se `dataset_episodios.json` já existe
2. Se existir, pergunta se você quer re-raspar ou usar o arquivo existente
3. Raspa cada saga com delay de 2 segundos entre requisições
4. Salva o resultado em `dataset_episodios.json` na mesma pasta

**Sagas coletadas:**
- Saga East Blue
- Saga Alabasta
- Saga Ilha do Céu
- Saga Water 7
- Saga Thriller Bark
- Saga Cúpula da Guerra
- Saga Ilha dos Homens-Peixe

> ℹ️ O scraping leva alguns minutos. Você verá o progresso no console.

---

## 💬 Iniciando o Chatbot

Após o scraping ter gerado o `dataset_episodios.json`, execute:

```bash
python main.py
```

A janela do chatbot abrirá automaticamente. O carregamento inicial (dataset + NLP) pode levar alguns segundos. Aguarde a mensagem de boas-vindas.

---

## 🗣️ Exemplos de Perguntas

### Filtros por tipo de episódio
```
Liste os episódios filler da Saga East Blue
Quais são os filmes da Saga Alabasta?
Mostre os episódios recap da Saga Thriller Bark
Quantos OVAs existem?
Episódios especial TV do Arco Loguetown
```

### Filtros por arco ou saga
```
Episódios do Arco Arlong Park
Listar episódios da Saga Water 7
Quais episódios pertencem ao Arco Baratie?
```

### Busca semântica (conteúdo livre)
```
Em qual episódio Luffy luta contra Arlong?
Me fala sobre o encontro de Zoro e Mihawk
Qual episódio mostra o passado de Nami?
Episódio onde Luffy usa Gear Second
Qual episódio tem a batalha de Marineford?
Quero ver algo sobre o chapéu de palha
```

### Perguntas gerais
```
Ver Arcos Disponíveis (botão na interface)
Quantos episódios existem no total?
```

---

## 📁 Estrutura de Arquivos

```
proc_ling_natural_projeto/
├── scraper.py              # Módulo de web scraping
├── nlp_engine.py           # Motor de NLP e classe OnePieceChatbot
├── main.py                 # Interface gráfica Tkinter
├── requirements.txt        # Dependências Python
├── README.md               # Este arquivo
└── dataset_episodios.json  # Gerado após executar o scraper
```

---

## 🏗️ Arquitetura

```
scraper.py          ──►  dataset_episodios.json
                                  │
                                  ▼
nlp_engine.py ◄── OnePieceChatbot(path='dataset_episodios.json')
      │                           │
      │              ┌────────────┴────────────┐
      │              ▼                         ▼
      │      Filtros lógicos          Busca semântica
      │      (pandas + regex)      (TF-IDF + cosseno)
      │
      ▼
main.py ──► JanelaChatbot (Tkinter + threads)
```

---

## 🎨 Interface

- **Tema escuro** (`#1e1e1e`)
- Mensagens do usuário em **azul** (`#4fc3f7`)
- Respostas do bot em **verde** (`#a5d6a7`)
- Botão **Enviar** em vermelho One Piece (`#e53935`)
- Suporte a `Enter` para envio rápido
- Botão **📋 Ver Arcos Disponíveis** para exploração

---

## 📦 Dependências Utilizadas

| Biblioteca | Uso |
|---|---|
| `requests` + `beautifulsoup4` | Web Scraping da Fandom Wiki |
| `spacy` + `pt_core_news_lg` | NER e análise linguística |
| `scikit-learn` | TF-IDF e similaridade de cossenos |
| `pandas` | Manipulação do dataset em DataFrame |
| `nltk` | Suporte auxiliar de tokenização |
| `tkinter` | Interface gráfica nativa |

---

## ⚠️ Solução de Problemas

**"Dataset não encontrado"**
→ Execute `python scraper.py` primeiro.

**"Modelo spaCy não encontrado"**
→ Execute `python -m spacy download pt_core_news_lg`.

**"Nenhum episódio encontrado" para perguntas válidas**
→ Tente usar o botão 📋 para ver os nomes exatos dos arcos e sagas disponíveis.

**Scraping falhou / não retornou episódios**
→ Verifique sua conexão com a internet. A Fandom Wiki pode estar instável.
→ Aguarde alguns minutos e tente novamente.