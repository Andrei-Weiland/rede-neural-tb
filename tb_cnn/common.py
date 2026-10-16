"""Funções compartilhadas: leitura DICOM, máscara total do pulmão e DenseNet121."""
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
import torch.nn as nn
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from torchvision import models

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "DIVISAO 80-10-10"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
MODELOS_DIR = Path(__file__).resolve().parent / "modelos"
MODELO_FINAL = MODELOS_DIR / "modelo_final.pt"

IMG_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TEXTO_POSITIVO = "parece tuberculose (sugestivo)"
TEXTO_NEGATIVO = "não parece tuberculose (não sugestivo)"


def ler_imagem(path):
    """Lê DICOM ou imagem comum e devolve uint8 em tons de cinza (pulmão escuro, osso claro)."""
    path = Path(path)
    if path.suffix.lower() == ".dcm":
        ds = pydicom.dcmread(path)
        arr = apply_voi_lut(apply_modality_lut(ds.pixel_array, ds), ds).astype(np.float32)
        if ds.PhotometricInterpretation == "MONOCHROME1":
            arr = arr.max() - arr
    else:
        arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError(f"Não foi possível ler a imagem: {path}")
        arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, (0.5, 99.5))
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def redimensionar(img):
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


_segmentador = None


def mascara_pulmao(img512):
    """Máscara (0/1) dos dois pulmões, dilatada para não cortar lesões na borda."""
    global _segmentador
    import torchxrayvision as xrv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _segmentador is None:
        _segmentador = xrv.baseline_models.chestx_det.PSPNet().eval().to(device)
    x = torch.from_numpy(xrv.datasets.normalize(img512.astype(np.float32), 255))[None, None].to(device)
    with torch.no_grad():
        saida = torch.sigmoid(_segmentador(x))[0].cpu()
    alvos = _segmentador.targets
    mascara = (saida[alvos.index("Left Lung")] > 0.5) | (saida[alvos.index("Right Lung")] > 0.5)
    return cv2.dilate(mascara.numpy().astype(np.uint8), np.ones((31, 31), np.uint8))


def mascara_total(img512, mascara):
    """Zera fora do pulmão, recorta a caixa pulmonar e redimensiona para 512x512.

    Devolve (recorte, caixa x0, y0, x1, y1 na imagem 512).
    """
    ys, xs = np.nonzero(mascara)
    if len(xs) < 0.05 * mascara.size:
        return img512, (0, 0, IMG_SIZE, IMG_SIZE)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return redimensionar((img512 * mascara)[y0:y1, x0:x1]), (x0, y0, x1, y1)


def para_tensor(img_uint8):
    """uint8 em tons de cinza -> tensor (3, H, W) em [0, 1]."""
    return torch.from_numpy(img_uint8).float().div(255).unsqueeze(0).repeat(3, 1, 1)


def normalizar(t):
    return (t - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)


def criar_modelo(pretrained=True):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def carregar_modelo(device, caminho=None):
    """Devolve (modelo, limiar) do modelo final do caso 2."""
    caminho = Path(caminho) if caminho else MODELO_FINAL
    ck = torch.load(caminho, map_location=device, weights_only=False)
    model = criar_modelo(pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), float(ck["limiar"])


def gradcam(model, x):
    """Probabilidade de TB e mapa de calor Grad-CAM (0 a 1) sobre a última camada convolucional."""
    guardado = {}

    def hook(_, __, saida):
        guardado["ativ"] = saida
        saida.register_hook(lambda g: guardado.__setitem__("grad", g))

    h = model.features.register_forward_hook(hook)
    model.zero_grad()
    logit = model(x).squeeze()
    logit.backward()
    h.remove()
    pesos = guardado["grad"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((pesos * guardado["ativ"]).sum(dim=1)).squeeze(0)
    return torch.sigmoid(logit).item(), (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
