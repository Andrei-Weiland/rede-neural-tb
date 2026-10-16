"""Analisa um exame (ou uma pasta) com o modelo final (máscara total + DenseNet121).

Uso:
    python predict.py caminho/do/exame.dcm
    python predict.py "caminho/da/pasta" --saida saida_predicoes
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from common import (IMG_SIZE, TEXTO_NEGATIVO, TEXTO_POSITIVO, carregar_modelo, gradcam, ler_imagem,
                    mascara_pulmao, mascara_total, normalizar, para_tensor, redimensionar)

EXTENSOES = {".dcm", ".png", ".jpg", ".jpeg"}
ALTURA_SAIDA = 1024
LIMIAR_REGIAO = 0.5


def analisar(img, model, device):
    """Probabilidade de TB e Grad-CAM na imagem 512x512, restrito ao pulmão."""
    img512 = redimensionar(img)
    mascara = mascara_pulmao(img512)
    recorte, (x0, y0, x1, y1) = mascara_total(img512, mascara)
    prob, cam = gradcam(model, normalizar(para_tensor(recorte)).unsqueeze(0).to(device))
    cheio = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    cheio[y0:y1, x0:x1] = cv2.resize(cam, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    cheio = np.clip(cheio, 0, 1) * mascara
    return prob, cheio / (cheio.max() + 1e-8)


def fonte(tamanho):
    for nome in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            continue
    return ImageFont.load_default()


def desenhar(img_cinza, cam, sugestivo):
    h, w = img_cinza.shape
    img = cv2.resize(img_cinza, (round(w * ALTURA_SAIDA / h), ALTURA_SAIDA), interpolation=cv2.INTER_AREA)
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if sugestivo:
        cam = np.clip(cv2.resize(cam, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC), 0, 1)
        cor = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        mascara = (cam >= LIMIAR_REGIAO).astype(np.uint8)
        base = np.where(mascara[..., None] > 0, cv2.addWeighted(base, 0.6, cor, 0.4, 0), base)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(base, contornos, -1, (255, 255, 255), 2, cv2.LINE_AA)

    pil = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    texto = TEXTO_POSITIVO if sugestivo else TEXTO_NEGATIVO
    f = fonte(28)
    x0, y0, x1, y1 = draw.multiline_textbbox((0, 0), texto, font=f, align="center")
    x = (pil.width - (x1 - x0)) // 2
    y = pil.height - (y1 - y0) - 30
    draw.rectangle((x - 15, y - 10, x + (x1 - x0) + 15, y + (y1 - y0) + 15), fill=(40, 40, 40))
    draw.multiline_text((x, y), texto, font=f, fill=(255, 80, 80) if sugestivo else (120, 230, 120), align="center")
    return pil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada", help="arquivo .dcm/.png/.jpg ou pasta com exames")
    ap.add_argument("--saida", default="saida_predicoes", help="pasta onde salvar as imagens marcadas")
    ap.add_argument("--modelo", default=None, help="caminho do .pt (padrão: modelos/modelo_final.pt)")
    args = ap.parse_args()

    entrada = Path(args.entrada)
    arquivos = sorted(p for p in entrada.iterdir() if p.suffix.lower() in EXTENSOES) if entrada.is_dir() else [entrada]
    arquivos = [p for p in arquivos if not p.stem.endswith("_lunit")]
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, limiar = carregar_modelo(device, args.modelo)

    linhas = []
    for arq in arquivos:
        img = ler_imagem(arq)
        prob, cam = analisar(img, model, device)
        sugestivo = prob >= limiar
        resultado = TEXTO_POSITIVO if sugestivo else TEXTO_NEGATIVO
        destino = saida / f"{arq.stem}_resultado.jpg"
        desenhar(img, cam, sugestivo).save(destino, quality=92)
        print(f"{arq.name}: {resultado}")
        linhas.append({"arquivo": arq.name, "resultado": resultado, "imagem": str(destino)})

    if len(linhas) > 1:
        pd.DataFrame(linhas).to_csv(saida / "resultados.csv", index=False, encoding="utf-8-sig")
        print(f"\n{len(linhas)} exames analisados. Tabela salva em {saida / 'resultados.csv'}")


if __name__ == "__main__":
    main()
