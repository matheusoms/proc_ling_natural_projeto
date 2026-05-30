# Documentação Técnica: Chatbot Especialista de One Piece (Motor Híbrido de PLN)

> **Documento de Especificação Técnica (Technical Specification Document)**
> Detalhamento da arquitetura do sistema, implementação dos módulos de Processamento de Linguagem Natural (PLN) e infraestrutura.

---

## 1. Arquitetura do Sistema

O sistema foi arquitetado como uma aplicação local assíncrona composta por três módulos principais independentes e fortemente coesos: extração de dados, motor de linguagem natural e interface de usuário.

### 1.1 Visão Geral da Stack
- **Linguagem:** Python 3.12+
- **Processamento de Linguagem Natural (PLN):** `spaCy` (lemmatização e normalização), `scikit-learn` (vetorização TF-IDF e Similaridade de Cossenos), `langdetect` (detecção de idioma) e `deep-translator` (tradução automática).
- **Inteligência Artificial (Machine Learning):** Hugging Face `transformers` com PyTorch (`nlptown/bert-base-multilingual-uncased-sentiment`).
- **Web Scraping:** `cloudscraper` (bypass anti-bot) e `beautifulsoup4` (parsing de DOM).
- **Manipulação de Dados:** `pandas` para estruturação e filtragem lógica do dataset.
- **Interface Gráfica (GUI):** `tkinter` nativo, com operações pesadas delegadas a threads secundárias para evitar travamento da renderização.

### 1.2 Fluxo de Dados e Comunicação
1. **Fase de Extração:** O arquivo `scraper.py` atua como um ETL (Extract, Transform, Load). Ele faz o scraping híbrido de páginas da Fandom Wiki, extraindo metadados e resumos de episódios, gerando o artefato de conhecimento estático `dataset_episodios.json`.
2. **Fase de Inicialização:** A execução de `main.py` dispara a instanciação assíncrona da classe `OnePieceChatbot` definida em `nlp_engine.py`. Esta classe lê o dataset JSON, carrega modelos pesados (Hugging Face e spaCy), computa a matriz TF-IDF baseada nos resumos pré-processados e fica pronta em memória.
3. **Fase de Execução (Ciclo de Conversa):** 
   - O usuário interage via interface (`main.py`).
   - A entrada é enviada em thread secundária para o método `gerar_resposta()` em `nlp_engine.py`.
   - Ocorre a etapa de PLN corporativo: *Detecção de Idioma* → *Análise de Sentimento (Humor)* → *Extração de Entidades / Classificação de Intenção* → *Roteamento (Filtros Lógicos vs Semântica)* → *Formatação e Tradução de Retorno*.
   - A resposta processada é devolvida e renderizada na GUI principal.

---

## 2. Implementação Detalhada dos Requisitos

Abaixo, detalha-se o mapeamento em código da segunda etapa do projeto.

### Requisito 1: Módulo de Idioma
A aplicação é capaz de interagir fluidamente em Português e Inglês, detectando o idioma dinamicamente sem necessidade de troca de configuração pelo usuário.
- **Detecção:** O método `detectar_idioma()` da classe `OnePieceChatbot` (`nlp_engine.py`) utiliza a biblioteca `langdetect`. A semente da fábrica é travada para determinismo (`DetectorFactory.seed = 0`).
- **Processamento de Respostas:** Se o usuário fala em Inglês (`idioma == 'en'`), as strings de resposta base do chatbot são traduzidas em runtime (tempo de execução).
- **Tradução Otimizada:** Foi implementado um fallback seguro e um cache em memória. O método `_inicializar_tradutor()` tenta instanciar `GoogleTranslator` do pacote `deep-translator`.
- **Implementação:** O método `_traduzir_se_necessario()` armazena o resultado em `self._cache_traducoes`. Isso evita chamadas sucessivas à API para o mesmo texto retornado, economizando requisições HTTP e acelerando o tempo de resposta do robô. A formatação do episódio (`_formatar_episodio`) se adapta à língua.

### Requisito 2: Gatilhamento de Sentimento
O sistema monitora o comportamento textual do usuário para inferir frustração, raiva ou alegria, aplicando regras de negócio e prevenindo falsos positivos para perguntas secas.
- **Extração de Sentimento:** Executado primariamente pelo modelo neural em `_classificar_humor_imediato()`, que converte os escores do Hugging Face (stars) em rótulos `positivo`, `neutro` e `negativo`.
- **Regras de Negócio e Limiares (Thresholds):** A função `analisar_humor()` (`nlp_engine.py`) foi implementada com uma proteção rigorosa. Antes de invocar o modelo ML, o sistema checa se a entrada é uma pergunta factual curta:
  - Condições de bypass: Começa com marcadores de interrogação (variável `PREFIXOS_INTERROGATIVOS`), tem menos de 10 palavras e **não** possui marcadores agressivos óbvios (variável `MARCADORES_NEGATIVOS`).
  - *Motivo:* Perguntas curtas como "onde o luffy nasceu?" geralmente caem como "negativas" por falta de semântica no modelo padrão. O bypass classifica-as manualmente como `neutro`, evitando comportamento excessivamente empático por engano.

### Requisito 3: Interface com o Usuário
Desenvolvida integralmente de forma nativa e agnóstica a SO via `tkinter` no arquivo `main.py`.
- **Renderização e Layout:** A classe `JanelaChatbot` controla os frames. O histórico de mensagens é gerenciado por um `scrolledtext.ScrolledText` em modo "somente-leitura".
- **Sistema de Cores (Theming):** Um dicionário constante `CORES` foi utilizado para manter o design em "Dark Mode" de forma uniforme (cores como `#1e1e1e` para o fundo, texto usuário em ciano `#4fc3f7`, bot em verde `#a5d6a7`). Tags virtuais de cor são injetadas no ScrolledText (`self._area_chat.tag_configure()`) para diferenciar usuário e sistema.
- **Multithreading para UX:** O carregamento do modelo de PLN leva tempo. Para não dar o aspecto de "congelamento" de software (*application not responding*), o método `_iniciar_carregamento()` despacha a carga do modelo da classe `OnePieceChatbot` via `threading.Thread`. Ao terminar, usa-se `self._raiz.after()` para comunicar de volta com a thread principal de UI. O mesmo ocorre no envio de mensagens (`_processar_e_responder()`).

### Requisito 4: Base de Conhecimento Otimizada
O conhecimento da aplicação é restrito a um JSON gerado offline para garantir respostas curadas e offline (focadas estritamente em arcos do anime antes do "Time-skip").
- **Coleta e Limpeza Inicial:** O `scraper.py` implementa duas lógicas de extração (formatos de tabela vs lista) contornando firewalls da web via `cloudscraper`.
- **Pré-processamento de Backend:** Quando o JSON é lido pelo pandas (`nlp_engine.py`), a base de conhecimento sofre sanitização via `_pre_processar_para_busca()`.
- **Pipeline de Limpeza (`spaCy`):** Todo o texto de 'resumo' de episódios passa pelo motor `pt_core_news_lg`. A função normaliza texto retirando acentos (`unicodedata`), processa o documento (`doc = self._nlp(...)`) e filtra cada token gerando o "lemma" (`t.lemma_`), descartando _stop words_ (`t.is_stop`) e pontuações (`t.is_punct`). O texto denso é transformado em um vetor limpo contendo apenas termos essenciais, otimizando muito a precisão do TF-IDF.

### Requisito 5: Chat de Adaptação de Humor
A modulação da resposta ocorre em três camadas a partir da inteligência emocional processada:
- **Estado de Sessão Progressivo:** Em vez de analisar cada frase isoladamente, o `nlp_engine.py` mantém um histórico com janela de tamanho flexível: `self._historico_humor: deque = deque(maxlen=20)`.
- **Média Ponderada (Decaimento Temporal):** No método `_obter_estado_sessao()`, o chatbot verifica as últimas 5 mensagens aplicando pesos (1 a 5, onde a mensagem mais recente tem peso maior). Se o score negativo ponderado atingir um limite agudo (`score_neg >= 0.60`), o estado muda para `extremamente_frustrado`.
- **Modulação do Estilo de Resposta:** No método `gerar_resposta()`, um dicionário mapeia os rótulos emocionais para prefixos empáticos apropriados, inclusive diferenciando pelo idioma base detectado. Por exemplo, detectando frustração, ele injeta frases humanizadas de acalmamento ("Sinto muito pela frustração", ou em inglês, "I'm sorry you feel that way...").
- Um botão na GUI ("🔄 Nova Conversa") dispara a função `reset_sessao()` para limpar a fila de estados.

### Requisito 6: Motor Híbrido de Respostas
O sistema roteia e responde de forma híbrida e otimizada (determinística vs probabilística):
- **Classificador de Intenção Rápido:** No `_detectar_intencao()`, ele checa se a string contém palavras-chave de filtro listadas em `KEYWORDS_FILTRO` (ex: "filler", "filme", "canon").
- **Abordagem 1 - Filtros Lógicos Determinísticos:** Se a intenção for mapeada como 'filtro', a aplicação delega ao `pandas` em `_busca_por_filtros()`. O motor varre a string por entidades (arcos, tipos) e filtra os DataFrames booleana e precisamente, listando os resultados (ex.: listar todos os fillers de uma saga).
- **Abordagem 2 - Similaridade de Cossenos (Busca Semântica):** Para perguntas de conteúdo, o método `_busca_semantica()` atua. O texto sofre o preprocessamento `spaCy` e é vetorizado por `self._vetorizador.transform()` baseado no treinamento inicial TF-IDF.
- **Randomização Guiada (Bônus de Personagens):** Após o cálculo da distância espacial por `cosine_similarity`, um sistema extra de impulsionamento foi construído. O método verifica presença de personagens (ex.: "luffy", "zoro") no prompt e injeta escores arbitrários diretamente nos arrays numpy dos índices que contém esses nomes explícitos no 'resumo' `sim[mask_resumo] += 0.25` e no 'titulo' `sim[mask_titulo] += 0.40`. Isso resolve empates no TF-IDF de forma direcionada, sempre oferecendo a melhor predição combinada. Se a similaridade global não exceder o limite basal de `0.15`, ativa-se o aviso de Fallback (não encontrou resultado).

### Requisito 7: Integração Hugging Face
A integração com o ecossistema de Modelos Grandes de Linguagem Open-Source dá vida ao motor de sentimento de forma estritamente local (sem custos de API).
- **Modelo Usado:** `nlptown/bert-base-multilingual-uncased-sentiment`. Um modelo de arquitetura BERT treinado multilinguamente com finetuning focado na detecção de "stars" (estrelas 1 a 5) em revisões (reviews).
- **Injeção da API Local:** Ocorre no construtor `__init__` da classe `OnePieceChatbot` (`nlp_engine.py`), através da API oficial de abstração de tarefas `pipeline("sentiment-analysis", ...)` do módulo `transformers`.
- **Implementação:** A rede neural entra em ação pontual no fluxo de `_classificar_humor_imediato()`. O tensor processa no máximo 512 tokens em cada requisição (`texto[:512]`) e os outputs nominais (e.g., "1 star") são normalizados nativamente no código para categorias lógicas padronizadas (`negativo`, `neutro`, `positivo`).

---

## 3. Guia de Configuração e Execução

Para levantar o ambiente de produção local do chatbot, siga os procedimentos abaixo:

### 3.1 Requisitos do Ambiente
- **Engine:** Python 3.12 (Aconselhável ambiente virtual `python -m venv .venv`).
- **RAM Mínima:** Pelo menos 2GB livres de RAM (modelos Transformers e spaCy são pesados na montagem inicial).
- **Disco:** ~1 GB de armazenamento para dependências e cache de modelos de Machine Learning.

### 3.2 Dependências (`requirements.txt`)
O projeto exige pacotes numéricos, de interface web scraping e pipelines neurais. As principais versões travadas (conforme o manifesto original) são:
```text
beautifulsoup4>=4.12.3
spacy>=3.7.4
scikit-learn>=1.4.2
pandas>=2.2.1
cloudscraper>=1.2.3
transformers>=4.38.0
torch>=2.2.0
langdetect>=1.0.9
deep-translator>=1.11.4
```

### 3.3 Passos de Instalação e Build
Abra um terminal (powershell ou bash) na raiz do projeto e siga:

**1. Instalar as bibliotecas essenciais e binários dependentes:**
```bash
pip install -r requirements.txt
```

**2. Baixar o Modelo de Idioma (Lexical/Sintático) PT-BR para o spaCy:**
Isso habilita a Lematização em português.
```bash
python -m spacy download pt_core_news_lg
```

**3. Gerar a Base de Conhecimento Offline (ETL Inicial):**
O sistema necessita do arquivo `dataset_episodios.json`. Para extrair os dados da web, rode o módulo de scraper isoladamente pela primeira vez.
```bash
python scraper.py
```
*(O script rodará e avisará sobre a raspagem das páginas da Fandom com delays progressivos configurados. Aguarde a conclusão e o aviso de sucesso.)*

**4. Executar o Chatbot com Interface de Usuário:**
Com os passos anteriores completos, basta inicializar o motor e a UI.
```bash
python main.py
```
A janela do sistema `tkinter` será invocada. Inicialmente o frame carregará um modelo PyTorch / Hugging Face na memória. Após o aviso do "Bot" no chat, você poderá realizar consultas semânticas e filtrar arcos dinamicamente.

---
*Gerado por Engenheiro de Software Sênior e Analista Técnico.*
