"""Utilitários de treino: dataset balanceado, DenseNet121, métricas e limiar de Youden."""
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from common import CACHE_DIR, criar_modelo, normalizar, para_tensor


class RaioX(Dataset):
    """No treino: oversampling da classe sem TB + data augmentation online em todas as imagens."""

    def __init__(self, df, pasta_cache, treino=False, seed=42):
        self.imgs = [cv2.imdecode(np.fromfile(str(CACHE_DIR / pasta_cache / a), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                     for a in df["arquivo"]]
        self.rotulos = df["rotulo"].to_numpy(dtype=np.float32)
        self.itens = [(i, treino) for i in range(len(self.imgs))]
        if treino:
            sem_tb = np.flatnonzero(self.rotulos == 0)
            faltam = int((self.rotulos == 1).sum()) - len(sem_tb)
            extras = np.random.default_rng(seed).choice(sem_tb, faltam, replace=faltam > len(sem_tb))
            self.itens += [(int(i), True) for i in extras]
        self.aug = v2.Compose([
            v2.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    def __len__(self):
        return len(self.itens)

    def __getitem__(self, k):
        i, aumentar = self.itens[k]
        x = para_tensor(self.imgs[i])
        if aumentar:
            x = self.aug(x)
        return normalizar(x), torch.tensor(self.rotulos[i])


@torch.no_grad()
def prever(model, df, pasta_cache, args, device):
    model.eval()
    probs = []
    for x, _ in DataLoader(RaioX(df, pasta_cache), batch_size=args.batch):
        with torch.autocast("cuda", dtype=torch.float16):
            probs.append(torch.sigmoid(model(x.to(device)).squeeze(1)).float().cpu())
    return torch.cat(probs).numpy()


def treinar(treino, val, pasta_cache, args, device):
    """Treina com parada antecipada pela AUC de validação e devolve o modelo da melhor época."""
    dl = DataLoader(RaioX(treino, pasta_cache, treino=True, seed=args.seed), batch_size=args.batch, shuffle=True,
                    drop_last=True)
    model = criar_modelo().to(device)
    criterio = nn.BCEWithLogitsLoss()
    otim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(otim, T_max=args.epocas)
    scaler = torch.amp.GradScaler()

    melhor_auc, sem_melhora, melhor_estado = -1.0, 0, None
    for _ in range(args.epocas):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            otim.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                perda = criterio(model(x).squeeze(1), y)
            scaler.scale(perda).backward()
            scaler.step(otim)
            scaler.update()
        sched.step()
        auc = roc_auc_score(val["rotulo"], prever(model, val, pasta_cache, args, device))
        if auc > melhor_auc:
            melhor_auc, sem_melhora = auc, 0
            melhor_estado = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= args.paciencia:
                break
    model.load_state_dict(melhor_estado)
    return model


def limiar_youden(y, p):
    fpr, tpr, cortes = roc_curve(y, p)
    return float(np.clip(cortes[np.argmax(tpr - fpr)], 0, 1))


def metricas(y, p, limiar):
    tn, fp, fn, tp = confusion_matrix(y, p >= limiar, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y, p),
        "acuracia": (tp + tn) / len(y),
        "sensibilidade": tp / max(tp + fn, 1),
        "especificidade": tn / max(tn + fp, 1),
    }


def resumo(lista):
    return {c: {"media": round(float(np.mean([m[c] for m in lista])), 4),
                "desvio": round(float(np.std([m[c] for m in lista], ddof=1)), 4)} for c in lista[0]}
