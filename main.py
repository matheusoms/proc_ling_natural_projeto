"""
main.py — Interface Gráfica Tkinter para o Chatbot One Piece

Janela de chat com tema escuro, carregamento assíncrono do chatbot em thread
separada e suporte a busca por filtros e busca semântica via NLP.

Autor: Projeto PLN — FATEC
"""

import threading
import tkinter as tk
from tkinter import scrolledtext, font

from nlp_engine import OnePieceChatbot


# ---------------------------------------------------------------------------
# Paleta de cores
# ---------------------------------------------------------------------------

CORES = {
    "fundo": "#1e1e1e",
    "fundo_chat": "#1e1e1e",
    "fundo_entrada": "#2d2d2d",
    "texto_geral": "#f0f0f0",
    "texto_usuario": "#4fc3f7",
    "texto_bot": "#a5d6a7",
    "texto_sistema": "#9e9e9e",
    "botao_enviar_bg": "#e53935",
    "botao_enviar_fg": "#ffffff",
    "botao_arcos_bg": "#37474f",
    "botao_arcos_fg": "#b0bec5",
    "borda": "#424242",
    "barra_titulo": "#121212",
}


# ---------------------------------------------------------------------------
# Classe da janela principal
# ---------------------------------------------------------------------------


class JanelaChatbot:
    """
    Interface gráfica principal do Chatbot One Piece usando Tkinter.

    Gerencia o layout da janela, o carregamento assíncrono do chatbot e
    a troca de mensagens entre usuário e bot.
    """

    def __init__(self, raiz: tk.Tk) -> None:
        """
        Inicializa a janela e dispara o carregamento do chatbot em uma thread.

        Parâmetros
        ----------
        raiz : tk.Tk
            Janela raiz do Tkinter.
        """
        self._raiz = raiz
        self._chatbot: OnePieceChatbot | None = None
        self._carregando = True

        self._configurar_janela()
        self._construir_layout()
        self._iniciar_carregamento()

    # -----------------------------------------------------------------------
    # Configuração da janela
    # -----------------------------------------------------------------------

    def _configurar_janela(self) -> None:
        """Configura propriedades gerais da janela raiz."""
        self._raiz.title("🏴‍☠️  Guia Especialista One Piece")
        self._raiz.geometry("850x620")
        self._raiz.minsize(680, 500)
        self._raiz.configure(bg=CORES["fundo"])
        self._raiz.resizable(True, True)

        # Centralizar na tela
        self._raiz.update_idletasks()
        largura = self._raiz.winfo_width()
        altura = self._raiz.winfo_height()
        x = (self._raiz.winfo_screenwidth() // 2) - (largura // 2)
        y = (self._raiz.winfo_screenheight() // 2) - (altura // 2)
        self._raiz.geometry(f"{largura}x{altura}+{x}+{y}")

        # Interceptar fechamento da janela
        self._raiz.protocol("WM_DELETE_WINDOW", self._ao_fechar)

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------

    def _construir_layout(self) -> None:
        """Cria e posiciona todos os widgets da interface."""
        # --- Barra de título personalizada ---
        frame_titulo = tk.Frame(self._raiz, bg=CORES["barra_titulo"], pady=10)
        frame_titulo.pack(fill=tk.X, side=tk.TOP)

        tk.Label(
            frame_titulo,
            text="🏴‍☠️   Guia Especialista One Piece",
            bg=CORES["barra_titulo"],
            fg=CORES["texto_geral"],
            font=("Segoe UI", 15, "bold"),
        ).pack()

        tk.Label(
            frame_titulo,
            text="Pergunte sobre episódios, arcos, sagas e mais!",
            bg=CORES["barra_titulo"],
            fg=CORES["texto_sistema"],
            font=("Segoe UI", 9),
        ).pack()

        # --- Área de chat ---
        frame_chat = tk.Frame(self._raiz, bg=CORES["fundo_chat"])
        frame_chat.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8, 0))

        self._area_chat = scrolledtext.ScrolledText(
            frame_chat,
            state=tk.DISABLED,
            wrap=tk.WORD,
            bg=CORES["fundo_chat"],
            fg=CORES["texto_geral"],
            font=("Consolas", 10),
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=CORES["borda"],
            padx=12,
            pady=8,
            cursor="arrow",
        )
        self._area_chat.pack(fill=tk.BOTH, expand=True)

        # Tags de cores para diferenciar tipos de mensagem
        self._area_chat.tag_configure(
            "usuario",
            foreground=CORES["texto_usuario"],
            font=("Consolas", 10, "bold"),
        )
        self._area_chat.tag_configure(
            "bot",
            foreground=CORES["texto_bot"],
            font=("Consolas", 10),
        )
        self._area_chat.tag_configure(
            "sistema",
            foreground=CORES["texto_sistema"],
            font=("Consolas", 9, "italic"),
        )
        self._area_chat.tag_configure(
            "separador",
            foreground=CORES["borda"],
        )

        # --- Frame da área de entrada ---
        frame_entrada = tk.Frame(self._raiz, bg=CORES["fundo"], pady=6, padx=10)
        frame_entrada.pack(fill=tk.X, side=tk.BOTTOM)

        # Linha inferior: campo de texto + botão Enviar
        frame_input_row = tk.Frame(frame_entrada, bg=CORES["fundo"])
        frame_input_row.pack(fill=tk.X)

        self._entrada = tk.Entry(
            frame_input_row,
            font=("Segoe UI", 11),
            bg=CORES["fundo_entrada"],
            fg=CORES["texto_geral"],
            insertbackground=CORES["texto_geral"],
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=CORES["borda"],
            state=tk.DISABLED,
        )
        self._entrada.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 6))

        self._botao_enviar = tk.Button(
            frame_input_row,
            text="Enviar ▶",
            command=self._ao_enviar,
            bg=CORES["botao_enviar_bg"],
            fg=CORES["botao_enviar_fg"],
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            padx=14,
            pady=6,
            activebackground="#c62828",
            activeforeground="#ffffff",
            state=tk.DISABLED,
        )
        self._botao_enviar.pack(side=tk.RIGHT)

        # Botão de arcos disponíveis
        self._botao_arcos = tk.Button(
            frame_entrada,
            text="📋  Ver Arcos Disponíveis",
            command=self._ao_ver_arcos,
            bg=CORES["botao_arcos_bg"],
            fg=CORES["botao_arcos_fg"],
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            cursor="hand2",
            pady=4,
            activebackground="#455a64",
            activeforeground=CORES["texto_geral"],
            state=tk.DISABLED,
        )
        self._botao_arcos.pack(fill=tk.X, pady=(4, 2))

        # Bind da tecla Enter ao envio
        self._raiz.bind("<Return>", lambda _: self._ao_enviar())

    # -----------------------------------------------------------------------
    # Carregamento assíncrono
    # -----------------------------------------------------------------------

    def _iniciar_carregamento(self) -> None:
        """Inicia o carregamento do chatbot em uma thread separada."""
        self._exibir_mensagem_sistema(
            "⏳ Carregando o dataset e inicializando o NLP... Por favor, aguarde."
        )

        thread = threading.Thread(target=self._carregar_chatbot, daemon=True)
        thread.start()

    def _carregar_chatbot(self) -> None:
        """
        Carrega a instância do OnePieceChatbot em uma thread separada.

        Ao concluir (com sucesso ou erro), agenda a atualização da UI na
        thread principal via after().
        """
        try:
            chatbot = OnePieceChatbot()
            # Agendar atualização na thread principal
            self._raiz.after(0, self._carregamento_concluido, chatbot)
        except FileNotFoundError as erro:
            self._raiz.after(0, self._carregamento_falhou, str(erro))
        except OSError as erro:
            self._raiz.after(0, self._carregamento_falhou, str(erro))
        except Exception as erro:
            self._raiz.after(0, self._carregamento_falhou, f"Erro inesperado: {erro}")

    def _carregamento_concluido(self, chatbot: OnePieceChatbot) -> None:
        """
        Chamado na thread principal após carregamento bem-sucedido do chatbot.

        Parâmetros
        ----------
        chatbot : OnePieceChatbot
            Instância pronta do chatbot.
        """
        self._chatbot = chatbot
        self._carregando = False

        # Habilitar widgets
        self._entrada.configure(state=tk.NORMAL)
        self._botao_enviar.configure(state=tk.NORMAL)
        self._botao_arcos.configure(state=tk.NORMAL)
        self._entrada.focus_set()

        # Mensagem de boas-vindas
        boas_vindas = (
            "🏴‍☠️  Olá! Sou o Guia Especialista de One Piece!\n\n"
            "Posso responder perguntas sobre episódios das sagas pré-time-skip.\n\n"
            "📌 Exemplos de perguntas:\n"
            "  • 'Liste os episódios filler do Arco Loguetown'\n"
            "  • 'Quais são os episódios recap da Saga Alabasta?'\n"
            "  • 'Quantos filmes existem na Saga East Blue?'\n"
            "  • 'Em qual episódio Luffy luta contra Arlong?'\n"
            "  • 'Me conta sobre o episódio com Zoro e Mihawk'\n"
            "  • 'Episódios da Saga Water 7'\n\n"
            "Use o botão 📋 para ver todos os arcos disponíveis!"
        )
        self._exibir_mensagem("bot", f"🤖 Bot: {boas_vindas}")

    def _carregamento_falhou(self, mensagem_erro: str) -> None:
        """
        Chamado na thread principal quando o carregamento falhou.

        Parâmetros
        ----------
        mensagem_erro : str
            Descrição do erro ocorrido.
        """
        self._carregando = False
        erro_formatado = (
            f"❌ Erro ao carregar o chatbot:\n{mensagem_erro}\n\n"
            "Verifique se o dataset existe e o modelo spaCy está instalado."
        )
        self._exibir_mensagem("sistema", erro_formatado)

    # -----------------------------------------------------------------------
    # Ações da interface
    # -----------------------------------------------------------------------

    def _ao_enviar(self) -> None:
        """
        Captura o texto do campo de entrada e processa a mensagem do usuário.

        Ignora mensagens vazias e exibe aviso se o chatbot ainda estiver
        carregando.
        """
        if self._carregando:
            return

        texto = self._entrada.get().strip()
        if not texto:
            return

        # Limpar campo de entrada
        self._entrada.delete(0, tk.END)

        # Exibir mensagem do usuário
        self._exibir_mensagem("usuario", f"🧑 Você: {texto}")

        # Gerar resposta em thread separada para não travar a UI
        thread = threading.Thread(
            target=self._processar_e_responder, args=(texto,), daemon=True
        )
        thread.start()

    def _processar_e_responder(self, texto: str) -> None:
        """
        Gera a resposta do chatbot em uma thread separada e agenda exibição.

        Parâmetros
        ----------
        texto : str
            Texto da mensagem do usuário.
        """
        try:
            resposta = self._chatbot.gerar_resposta(texto)
        except Exception as erro:
            resposta = (
                f"⚠️  Ocorreu um erro ao processar sua pergunta: {erro}\n"
                "Tente reformular sua pergunta."
            )

        # Agendar exibição na thread principal
        self._raiz.after(0, self._exibir_mensagem, "bot", f"🤖 Bot: {resposta}")

    def _ao_ver_arcos(self) -> None:
        """Exibe a listagem de arcos disponíveis no chat."""
        if self._chatbot is None or self._carregando:
            return

        thread = threading.Thread(target=self._buscar_e_exibir_arcos, daemon=True)
        thread.start()

    def _buscar_e_exibir_arcos(self) -> None:
        """Busca a listagem de arcos em thread separada e agenda exibição."""
        try:
            listagem = self._chatbot.listar_arcos()
        except Exception as erro:
            listagem = f"⚠️  Erro ao listar arcos: {erro}"

        self._raiz.after(
            0, self._exibir_mensagem, "bot", f"🤖 Bot: {listagem}"
        )

    def _ao_fechar(self) -> None:
        """Finaliza a aplicação ao fechar a janela."""
        self._raiz.destroy()

    # -----------------------------------------------------------------------
    # Exibição de mensagens no chat
    # -----------------------------------------------------------------------

    def _exibir_mensagem(self, tipo: str, texto: str) -> None:
        """
        Adiciona uma mensagem colorida à área de chat.

        Parâmetros
        ----------
        tipo : str
            Tipo da mensagem: 'usuario', 'bot' ou 'sistema'.
        texto : str
            Conteúdo da mensagem a exibir.
        """
        self._area_chat.configure(state=tk.NORMAL)

        # Separador visual entre mensagens
        self._area_chat.insert(tk.END, "\n", "separador")

        # Inserir texto com a tag de cor correspondente
        self._area_chat.insert(tk.END, texto + "\n", tipo)

        self._area_chat.configure(state=tk.DISABLED)
        self._area_chat.see(tk.END)

    def _exibir_mensagem_sistema(self, texto: str) -> None:
        """
        Exibe uma mensagem de sistema (informativa, em cinza itálico).

        Parâmetros
        ----------
        texto : str
            Mensagem informativa do sistema.
        """
        self._exibir_mensagem("sistema", texto)


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------


def main() -> None:
    """Inicializa e executa a aplicação Tkinter do Chatbot One Piece."""
    raiz = tk.Tk()
    aplicacao = JanelaChatbot(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
