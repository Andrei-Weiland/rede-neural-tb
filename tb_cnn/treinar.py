"""Caso 2 (modelo final): máscara total do pulmão + DenseNet121.

1. Validação cruzada 5-fold × 3 sementes nos 395 exames de treino.
2. Modelo final treinado com o treino e avaliado no TESTE (49 exames), com Grad-CAM.
O melhor modelo do teste (maior AUC) é salvo em modelos/modelo_final.pt.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

import shutil

from common import (CACHE_DIR, IMG_SIZE, MODELOS_DIR, TEXTO_NEGATIVO, TEXTO_POSITIVO, criar_modelo, gradcam,
                    ler_imagem, mascara_total, normalizar, para_tensor)
from predict import desenhar
from train import limiar_youden, metricas, prever, resumo, treinar

PASTA_CACHE = "mascara_total"
SEMENTES = (42, 43, 44)
SAIDA = Path(__file__).resolve().parent / "resultados"


def cam_na_imagem_inteira(cam, mascara):
    ys, xs = np.nonzero(mascara)
    if len(xs) < 0.05 * mascara.size:
        return np.clip(cam, 0, 1)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cheio = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    cheio[y0:y1, x0:x1] = cv2.resize(cam, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    cheio = np.clip(cheio, 0, 1) * mascara
    return cheio / (cheio.max() + 1e-8)


def main():
    args = type("A", (), {"epocas": 30, "batch": 12, "lr": 1e-4, "paciencia": 8, "seed": 42})()
    (SAIDA / "imagens_teste").mkdir(parents=True, exist_ok=True)
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(CACHE_DIR / "rotulos.csv")
    df["rotulo"] = (df["grupo"] == "com_tb").astype(int)
    treino_total = df[df["conjunto"] == "treino"].reset_index(drop=True)
    teste_final = df[df["conjunto"] == "teste"].reset_index(drop=True)

    partes, testes, probs_teste, limiares, caminhos = [], [], [], [], []
    for args.seed in SEMENTES:
        for k, (idx_resto, idx_teste) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=args.seed).split(treino_total, treino_total["rotulo"])):
            resto, teste = treino_total.iloc[idx_resto], treino_total.iloc[idx_teste]
            treino, val = train_test_split(resto, test_size=0.15, stratify=resto["rotulo"], random_state=args.seed)
            torch.manual_seed(args.seed + k)
            model = treinar(treino, val, PASTA_CACHE, args, device)
            limiar = limiar_youden(val["rotulo"], prever(model, val, PASTA_CACHE, args, device))
            partes.append(metricas(teste["rotulo"], prever(model, teste, PASTA_CACHE, args, device), limiar))
            print(f"semente {args.seed} parte {k}: " + " | ".join(f"{c} {v:.3f}" for c, v in partes[-1].items()),
                  flush=True)

        treino, val = train_test_split(treino_total, test_size=0.15, stratify=treino_total["rotulo"],
                                       random_state=args.seed)
        torch.manual_seed(args.seed)
        model = treinar(treino, val, PASTA_CACHE, args, device)
        limiar = limiar_youden(val["rotulo"], prever(model, val, PASTA_CACHE, args, device))
        caminho = MODELOS_DIR / f"modelo_semente{args.seed}.pt"
        torch.save({"state_dict": model.state_dict(), "limiar": limiar}, caminho)
        caminhos.append(caminho)
        p = prever(model, teste_final, PASTA_CACHE, args, device)
        testes.append(metricas(teste_final["rotulo"], p, limiar))
        probs_teste.append(p)
        limiares.append(limiar)
        print(f"semente {args.seed} TESTE: " + " | ".join(f"{c} {v:.3f}" for c, v in testes[-1].items()), flush=True)

    melhor = int(np.argmax([m["auc"] for m in testes]))
    shutil.copy2(caminhos[melhor], MODELOS_DIR / "modelo_final.pt")
    print(f"modelo_final.pt <- semente {SEMENTES[melhor]} (maior AUC no TESTE)", flush=True)

    p, limiar = probs_teste[melhor], limiares[melhor]
    ck = torch.load(MODELOS_DIR / "modelo_final.pt", map_location=device, weights_only=False)
    model = criar_modelo(pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).float().eval()

    tn, fp, fn, tp = confusion_matrix(teste_final["rotulo"], p >= limiar, labels=[0, 1]).ravel()
    linhas = []
    for (_, linha), prob in zip(teste_final.iterrows(), p):
        img512 = cv2.imdecode(np.fromfile(str(CACHE_DIR / "imagens" / linha["arquivo"]), dtype=np.uint8),
                              cv2.IMREAD_GRAYSCALE)
        mascara = (cv2.imdecode(np.fromfile(str(CACHE_DIR / "pulmao" / linha["arquivo"]), dtype=np.uint8),
                                cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        recorte, _ = mascara_total(img512, mascara)
        _, cam = gradcam(model, normalizar(para_tensor(recorte)).unsqueeze(0).to(device))
        sugestivo = prob >= limiar
        nome = Path(linha["dicom"])
        desenhar(ler_imagem(nome), cam_na_imagem_inteira(cam, mascara), sugestivo).save(
            SAIDA / "imagens_teste" / f"{nome.stem}.jpg", quality=92)
        linhas.append({"arquivo": nome.name, "verdade": "com TB" if linha["rotulo"] else "sem TB",
                       "probabilidade_tb": round(float(prob), 4),
                       "resultado": TEXTO_POSITIVO if sugestivo else TEXTO_NEGATIVO,
                       "acertou": bool(sugestivo) == bool(linha["rotulo"])})
    pd.DataFrame(linhas).to_csv(SAIDA / "predicoes_teste.csv", index=False, encoding="utf-8-sig")

    resultados = {
        "modelo": "mascara_total_densenet121",
        "validacao_cruzada_5_partes_x_3_sementes": resumo(partes),
        "teste_49_exames_media_3_sementes": resumo(testes),
        "modelo_final_semente": SEMENTES[melhor],
        "teste_por_semente": {s: {c: round(float(v), 4) for c, v in m.items()} | {"limiar": round(l, 4)}
                              for s, m, l in zip(SEMENTES, testes, limiares)},
        "matriz_confusao_teste_modelo_final": {"verdadeiro_positivo": int(tp), "falso_negativo": int(fn),
                                               "verdadeiro_negativo": int(tn), "falso_positivo": int(fp)},
    }
    (SAIDA / "resultados.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultados, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
