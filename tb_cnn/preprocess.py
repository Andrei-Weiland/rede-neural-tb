"""Converte os DICOM de DIVISAO 80-10-10 em PNG 512x512 com máscara total do pulmão.

Os arquivos originais são apenas lidos; tudo que é gerado fica em tb_cnn/cache.
"""
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import CACHE_DIR, DATA_DIR, ler_imagem, mascara_pulmao, mascara_total, redimensionar

PASTAS_TREINO = {
    "COM TUBERCULOSE": "com_tb",
    "SEM TUBERCULOSE": "sem_tb",
    "TB SCORE 25 OU MENOS (desconsiderada no treinamento)": "tb_25_ou_menos",
}
PASTAS_AVALIACAO = {
    "TESTE": "teste",
    "VALIDACAO ESPECIALISTAS": "especialistas",
}
PASTA_MASCARA = "mascara_total"


def listar_exames():
    for pasta, grupo in PASTAS_TREINO.items():
        for dcm in sorted((DATA_DIR / "TREINAMENTO" / pasta).glob("*.dcm")):
            yield dcm, grupo, "fora" if grupo == "tb_25_ou_menos" else "treino"
    grupos = pd.read_csv(DATA_DIR / "lista_movidos.csv", encoding="utf-8-sig").set_index("arquivo_dicom")["grupo"]
    for pasta, conjunto in PASTAS_AVALIACAO.items():
        for dcm in sorted((DATA_DIR / pasta).glob("*.dcm")):
            yield dcm, grupos[dcm.name], conjunto


def ler(caminho):
    return cv2.imdecode(np.fromfile(str(caminho), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def salvar(img, caminho):
    cv2.imencode(".png", img)[1].tofile(str(caminho))


def main():
    for pasta in ["imagens", "pulmao", PASTA_MASCARA]:
        (CACHE_DIR / pasta).mkdir(parents=True, exist_ok=True)

    linhas = []
    for dcm, grupo, conjunto in tqdm(list(listar_exames()), desc="exames"):
        nome = f"{dcm.stem}.png"
        if not (CACHE_DIR / "imagens" / nome).exists():
            salvar(redimensionar(ler_imagem(dcm)), CACHE_DIR / "imagens" / nome)
        if not (CACHE_DIR / "pulmao" / nome).exists():
            salvar(mascara_pulmao(ler(CACHE_DIR / "imagens" / nome)) * 255, CACHE_DIR / "pulmao" / nome)
        if not (CACHE_DIR / PASTA_MASCARA / nome).exists():
            img = ler(CACHE_DIR / "imagens" / nome)
            mascara = (ler(CACHE_DIR / "pulmao" / nome) > 0).astype(np.uint8)
            salvar(mascara_total(img, mascara)[0], CACHE_DIR / PASTA_MASCARA / nome)
        linhas.append({"arquivo": nome, "dicom": str(dcm), "grupo": grupo, "conjunto": conjunto})

    df = pd.DataFrame(linhas)
    df.to_csv(CACHE_DIR / "rotulos.csv", index=False)
    print(pd.crosstab(df["grupo"], df["conjunto"], margins=True, margins_name="total").to_string())


if __name__ == "__main__":
    main()
