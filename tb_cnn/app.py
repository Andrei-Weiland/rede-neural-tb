"""App de desktop: escolher um raio-x → analisar com o modelo → ver o resultado.

Uso:
    python app.py
"""
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image, ImageTk

from common import MODELO_FINAL, TEXTO_POSITIVO, carregar_modelo
from predict import analisar_arquivo

EXTENSOES = [("Exames", "*.dcm *.png *.jpg *.jpeg"), ("Todos", "*.*")]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Triagem de tuberculose")
        self.geometry("780x720")
        self.minsize(640, 560)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.limiar = None
        self.caminho = None
        self.foto = None

        titulo = ttk.Label(self, text="Triagem de tuberculose em raio-x", font=("Segoe UI", 16, "bold"))
        titulo.pack(pady=(16, 4))
        ttk.Label(self, text="Escolha um exame (.dcm, .png, .jpg) para analisar.", font=("Segoe UI", 10)).pack()

        barra = ttk.Frame(self)
        barra.pack(fill="x", padx=16, pady=12)
        self.lbl_arquivo = ttk.Label(barra, text="Nenhum arquivo selecionado", foreground="#555")
        self.lbl_arquivo.pack(side="left", fill="x", expand=True)
        ttk.Button(barra, text="Escolher arquivo", command=self.escolher).pack(side="right", padx=(8, 0))
        self.btn_analisar = ttk.Button(barra, text="Analisar", command=self.analisar, state="disabled")
        self.btn_analisar.pack(side="right")

        self.lbl_resultado = ttk.Label(self, text="", font=("Segoe UI", 14, "bold"))
        self.lbl_resultado.pack(pady=(4, 8))

        self.lbl_imagem = ttk.Label(self, anchor="center", background="#111")
        self.lbl_imagem.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        ttk.Label(
            self,
            text="Ferramenta de apoio à triagem. Não substitui avaliação médica.",
            font=("Segoe UI", 8),
            foreground="#777",
        ).pack(pady=(0, 10))

        self.after(100, self.carregar_modelo)

    def carregar_modelo(self):
        def tarefa():
            try:
                model, limiar = carregar_modelo(self.device, MODELO_FINAL)
                self.model, self.limiar = model, limiar
                self.after(0, lambda: self.status.config(text=f"Modelo pronto ({self.device})."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Erro", f"Não foi possível carregar o modelo:\n{e}"))
                self.after(0, lambda: self.status.config(text="Erro ao carregar o modelo."))

        threading.Thread(target=tarefa, daemon=True).start()

    def escolher(self):
        caminho = filedialog.askopenfilename(title="Escolher exame", filetypes=EXTENSOES)
        if not caminho:
            return
        self.caminho = Path(caminho)
        self.lbl_arquivo.config(text=self.caminho.name)
        self.btn_analisar.config(state="normal")
        self.lbl_resultado.config(text="")
        self.lbl_imagem.config(image="")
        self.foto = None

    def analisar(self):
        if self.model is None:
            messagebox.showinfo("Aguarde", "O modelo ainda está carregando.")
            return
        if self.caminho is None:
            return
        self.btn_analisar.config(state="disabled")
        self.status.config(text="Analisando…")
        caminho = self.caminho

        def tarefa():
            try:
                resultado, imagem = analisar_arquivo(caminho, self.model, self.limiar, self.device)
                self.after(0, lambda: self.mostrar(resultado, imagem))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Erro", f"Falha na análise:\n{e}"))
                self.after(0, lambda: self.status.config(text="Erro na análise."))
                self.after(0, lambda: self.btn_analisar.config(state="normal"))

        threading.Thread(target=tarefa, daemon=True).start()

    def mostrar(self, resultado, imagem: Image.Image):
        cor = "#c0392b" if resultado == TEXTO_POSITIVO else "#1e8449"
        self.lbl_resultado.config(text=resultado, foreground=cor)

        area_w = max(self.lbl_imagem.winfo_width(), 500)
        area_h = max(self.lbl_imagem.winfo_height(), 400)
        img = imagem.copy()
        img.thumbnail((area_w - 20, area_h - 20), Image.Resampling.LANCZOS)
        self.foto = ImageTk.PhotoImage(img)
        self.lbl_imagem.config(image=self.foto)
        self.status.config(text=f"Concluído: {self.caminho.name}")
        self.btn_analisar.config(state="normal")


if __name__ == "__main__":
    App().mainloop()
