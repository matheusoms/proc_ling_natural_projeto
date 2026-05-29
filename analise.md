# Auditoria Técnica e Plano de Integração de IA - Chatbot One Piece
**Branch:** `teste/correcao-chatbot`

## 1. Status Atual dos Requisitos (Atualizado)

| Requisito | Status | Observações |
| :--- | :--- | :--- |
| **1. Detecção de Idioma** | ✅ Implementado | Utiliza `langdetect` para classificar entre 'pt' e 'en', alterando a língua do prefixo da resposta. |
| **2. Intervenção Ativa de Sentimento** | ✅ Implementado | Utiliza pipeline de `sentiment-analysis` para identificar frustração e acolher o usuário na resposta. |
| **3. Interface Gráfica (GUI)** | ✅ Implementado | Construída com `tkinter`, layout responsivo e processamento assíncrono. |
| **4. Base de Conhecimento Própria** | ⚠️ Parcialmente Implementado | Utiliza `dataset_episodios.json` vetorizado via `TfidfVectorizer`. Falta evoluir para Embeddings Densos reais para um verdadeiro RAG. |
| **5. Adaptação Dinâmica de Humor** | ⚠️ Parcialmente Implementado | O bot ajusta o tom da resposta atual baseado no sentimento da mensagem (prefixos), mas não rastreia o histórico emocional *contínuo* ao longo da sessão. |
| **6. Arquitetura Híbrida de Resposta** | ⚠️ Parcialmente Implementado | Mescla respostas lógicas (filtros) e semânticas (cossenos), mas a resposta final é engessada (formatação rígida) e não passa por um modelo Gerativo (LLM). |
| **7. Integração Hugging Face** | ✅ Implementado | Utiliza a biblioteca `transformers` localmente para carregar o modelo `nlptown/bert-base-multilingual-uncased-sentiment`. |

---

## 2. Prós e Contras da Arquitetura Atual

### 👍 Prós
- **Evolução Rápida:** A branch `teste/correcao-chatbot` introduziu IA real usando Hugging Face (`transformers`).
- **Resiliência de Idioma e Humor:** A integração do `langdetect` combinada com o `sentiment-analysis` melhora drasticamente a experiência do usuário, tornando o bot mais empático.
- **Assincronicidade e Threading:** O motor pesado do Hugging Face e TF-IDF roda em background sem congelar a interface do Tkinter.

### 👎 Contras
- **Dependência do TF-IDF:** A busca por cossenos no TF-IDF é léxica. Perguntas com sinônimos que não existam exatamente nos resumos irão falhar, diferente de um *Embedding Vectorial*.
- **Ausência de Memória de Sessão:** O estado emocional e o contexto da conversa são reiniciados a cada nova mensagem.
- **Falta de Geração Natural:** As respostas ainda são montadas concatenando strings fixas do JSON, e não re-sintetizadas organicamente como um LLM moderno faria.

---

## 3. Documentação Funcional: Fluxo de Dados

1. **Inicialização (`main.py` e `nlp_engine.py`):**
   A classe carrega o dataset via Pandas, a NLP básica pelo `spaCy`, ajusta a matriz do `TfidfVectorizer` e inicializa o pipeline de `sentiment-analysis` do Hugging Face.

2. **Entrada e Pré-processamento:**
   - O texto do usuário passa pelo `langdetect` (função `detectar_idioma`) definindo se a interação será em PT ou EN.
   - Passa pelo modelo do Hugging Face (função `analisar_humor`) que pontua a frase de 1 a 5 estrelas, classificando como Positivo, Negativo ou Neutro.

3. **Análise de Intenção e Busca:**
   - Verifica presença de palavras-chave ('filtro' vs 'semantica').
   - Retorna os dados crus (DataFrame filtrado ou top-1 do Cosseno TF-IDF).

4. **Mescla e Resposta Final:**
   - O código seleciona um dicionário de "prefixos" empáticos correspondentes à junção de Idioma + Humor.
   - Concatena o prefixo com a resposta base de dados e exibe na GUI.

---

## 4. Guia de Integração (Plano de Ação para Itens Faltantes)

Para finalizar os requisitos que estão "Parcialmente Implementados" (4, 5 e 6), devemos transitar a base de busca para vetores densos e acoplar um LLM.

### A) Evoluindo a Base de Conhecimento para Dense Embeddings (Req 4)
Para entender sinônimos melhor que o TF-IDF, melhorando a precisão do RAG.

**No `nlp_engine.py`:**
```python
# Instale a biblioteca: pip install sentence-transformers
from sentence_transformers import SentenceTransformer, util

# No __init__:
self._embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
textos_para_indexar = self._df["resumo"].tolist()
self._corpus_embeddings = self._embedder.encode(textos_para_indexar, convert_to_tensor=True)

# Método de busca semântica densa:
def _busca_semantica_densa(self, texto: str) -> dict:
    query_emb = self._embedder.encode(texto, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, self._corpus_embeddings, top_k=1)[0]
    
    if hits[0]['score'] < 0.3: # Limiar semântico configurável
        return None
        
    return self._df.iloc[hits[0]['corpus_id']].to_dict()
```

### B) Rastreio Contínuo de Humor (Req 5)
Adicionar um histórico real para adaptar a conversa de forma inteligente ao longo do tempo.

**No `nlp_engine.py`:**
```python
# No __init__ adicione:
self._historico_humor = []

# Atualize a função analisar_humor para considerar o longo prazo:
def analisar_humor(self, texto: str) -> str:
    res = self._analisador_sentimento(texto[:512])[0]
    stars = int(res['label'].split(' ')[0])
    
    estado = "negativo" if stars <= 2 else "positivo" if stars >= 4 else "neutro"
    
    # Guarda o estado atual no histórico da classe
    self._historico_humor.append(estado)
    
    # Adaptação Dinâmica: Usuário frustrado consecutivamente
    if len(self._historico_humor) >= 3 and all(h == "negativo" for h in self._historico_humor[-3:]):
        return "extremamente_frustrado" # Dispara intervenções ainda mais acolhedoras
        
    return estado
```

### C) Arquitetura Híbrida Gerativa com LLM (Req 6)
Em vez de concatenar strings hardcoded, passar os dados da base de conhecimento (RAG) para uma API LLM responder de forma natural. 

```python
# Para testes, sugerimos a API do Groq (llama-3 gratuito e muito rápido) ou OpenAI.
import requests

def gerar_texto_natural(contexto: dict, pergunta: str, idioma: str, humor: str) -> str:
    prompt = f"""
    Você é um Guia Especialista em One Piece acolhedor.
    O usuário perguntou: "{pergunta}"
    O idioma é {idioma} e o humor atual do usuário é: {humor}.
    
    BASE DE CONHECIMENTO:
    Título: {contexto['titulo']}
    Resumo: {contexto['resumo']}
    
    Instruções: Utilize SOMENTE os fatos da Base de Conhecimento para montar uma resposta amigável, fluída e orgânica. Não invente fatos.
    """
    
    # Chamada real de API de LLM via OpenAI Client ou requests:
    # response = llm_client.chat.completions.create(...)
    # return response.choices[0].message.content
    return prompt
```