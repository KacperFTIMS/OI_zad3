"""
Analiza GLOBALNA modeli klasyfikacyjnych metodami XAI.

W odróżnieniu od pojedynczych objaśnień lokalnych (jeden obraz -> jedno wyjaśnienie),
ten skrypt generuje materiał pokazujący, jak model rozpoznaje DANĄ KLASĘ "globalnie",
tzn. na podstawie wielu przykładów dla każdej z klas. Powstają dwa typy ilustracji:

  1. SIATKI (grids)        - wiele przykładów (kolumny) x wszystkie klasy (wiersze),
                             pokazują powtarzalność wzorca atrybucji w obrębie klasy.
  2. MAPY UŚREDNIONE (mean) - jedna "globalna sygnatura" klasy = średnia mapa atrybucji
                             po wielu poprawnie sklasyfikowanych próbkach danej klasy.

Modele:
  - MNIST CNN          : Saliency (siatka + uśrednienie), LIME-SLIC (siatka).
  - MNIST MLP zoning16 : Feature Ablation 4x4 (siatka + uśrednienie).
  - Imagenette CNN     : LIME-SLIC (siatka per klasa) + Saliency uśredniony per klasa.

Uruchomienie:
  python explain_global.py            # pełny przebieg
  python explain_global.py smoke      # szybki test (mało próbek/klas)
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
from captum.attr import Saliency, Lime, FeatureAblation
from skimage.segmentation import slic

from cnn_models import MnistCnnStandard, ImagenetteCnnStandard
from cnn_data import get_dataset
from dataloader import load_mnist
from yep import MLP

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "wyniki_xai")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Stałe normalizacji
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
IMAGENETTE_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENETTE_STD = np.array([0.229, 0.224, 0.225])

IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
]

# Parametry (nadpisywane w trybie smoke)
N_GRID = 5        # liczba przykładów na klasę w siatkach
N_MEAN = 60       # liczba próbek do uśrednienia map per klasa
LIME_SAMPLES = 200
SLIC_MNIST = dict(n_segments=16, compactness=0.1, start_label=0, channel_axis=None)
SLIC_IMAGENETTE = dict(n_segments=50, compactness=10, start_label=0)


# ----------------------------------------------------------------------------
# Pomocnicze
# ----------------------------------------------------------------------------
def denorm_mnist(t):
    return t.squeeze().cpu().detach().numpy() * MNIST_STD + MNIST_MEAN


def denorm_imagenette(t):
    img = t.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    img = img * IMAGENETTE_STD + IMAGENETTE_MEAN
    return np.clip(img, 0, 1)


def find_correct_per_class(model, dataset, device, num_classes, n_needed,
                           max_scan=None, batch_size=256):
    """Zwraca dict {klasa: [indeksy poprawnie sklasyfikowanych próbek]} (do n_needed na klasę)."""
    model.eval()
    found = {c: [] for c in range(num_classes)}
    remaining = num_classes * n_needed
    n = len(dataset) if max_scan is None else min(max_scan, len(dataset))

    batch_imgs, batch_idx = [], []

    def flush():
        nonlocal remaining
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            preds = model(x).argmax(dim=1).cpu().numpy()
        for j, gi in enumerate(batch_idx):
            lab = int(dataset[gi][1])
            if preds[j] == lab and len(found[lab]) < n_needed:
                found[lab].append(gi)
                remaining -= 1
        batch_imgs.clear()
        batch_idx.clear()

    for i in range(n):
        img, _ = dataset[i]
        batch_imgs.append(img)
        batch_idx.append(i)
        if len(batch_imgs) >= batch_size:
            flush()
            if remaining <= 0:
                break
    flush()
    return found


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Zapisano: {os.path.relpath(path, PROJECT_ROOT)}")


# ----------------------------------------------------------------------------
# MNIST CNN — Saliency
# ----------------------------------------------------------------------------
def saliency_map(model, image_tensor, target):
    sal = Saliency(model)
    inp = image_tensor.clone().requires_grad_()
    attr = sal.attribute(inp, target=target)
    return np.abs(attr.squeeze().cpu().detach().numpy())


def mnist_cnn_saliency(model, dataset, device):
    print("[MNIST CNN] Saliency (siatka + uśrednienie)")
    num_classes = 10
    idxs = find_correct_per_class(model, dataset, device, num_classes, N_MEAN)

    # --- Siatka: 10 klas (wiersze) x N_GRID przykładów (kolumny) ---
    fig, axes = plt.subplots(num_classes, N_GRID, figsize=(N_GRID * 1.5, num_classes * 1.5))
    for c in range(num_classes):
        for k in range(N_GRID):
            ax = axes[c][k]
            ax.axis("off")
            if k < len(idxs[c]):
                img, _ = dataset[idxs[c][k]]
                it = img.unsqueeze(0).to(device)
                sal = saliency_map(model, it, c)
                ax.imshow(denorm_mnist(it), cmap="gray")
                ax.imshow(sal, cmap="jet", alpha=0.55)
            if k == 0:
                ax.set_title(f"klasa {c}", loc="left", fontsize=9)
    fig.suptitle("MNIST CNN — Saliency dla wielu przykładów każdej klasy", fontsize=12)
    save_fig(fig, "mnist_cnn_saliency_grid_all.png")

    # --- Uśrednione mapy: 2x5 ---
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for c in range(num_classes):
        ax = axes[c // 5][c % 5]
        ax.axis("off")
        if idxs[c]:
            acc = None
            for gi in idxs[c]:
                img, _ = dataset[gi]
                sal = saliency_map(model, img.unsqueeze(0).to(device), c)
                acc = sal if acc is None else acc + sal
            acc /= len(idxs[c])
            ax.imshow(acc, cmap="hot")
        ax.set_title(f"cyfra {c}  (n={len(idxs[c])})", fontsize=9)
    fig.suptitle("MNIST CNN — uśredniona mapa Saliency per klasa (globalna sygnatura)", fontsize=12)
    save_fig(fig, "mnist_cnn_saliency_mean_all.png")


# ----------------------------------------------------------------------------
# MNIST CNN — LIME
# ----------------------------------------------------------------------------
def lime_mnist_attr(model, image_tensor, target):
    image_np = image_tensor.squeeze(0).squeeze(0).cpu().detach().numpy()
    segments = slic(image_np, **SLIC_MNIST)
    fmask = torch.tensor(segments).unsqueeze(0).unsqueeze(0).to(image_tensor.device)
    baselines = image_tensor * 0.0 + (-MNIST_MEAN / MNIST_STD)  # tło w przestrzeni znormalizowanej
    lime = Lime(model)
    attr = lime.attribute(image_tensor, target=target, baselines=baselines,
                          feature_mask=fmask, n_samples=LIME_SAMPLES)
    return attr.squeeze(0).squeeze(0).cpu().detach().numpy()


def mnist_cnn_lime(model, dataset, device):
    print("[MNIST CNN] LIME-SLIC (siatka)")
    num_classes = 10
    idxs = find_correct_per_class(model, dataset, device, num_classes, N_GRID)
    fig, axes = plt.subplots(num_classes, N_GRID, figsize=(N_GRID * 1.5, num_classes * 1.5))
    for c in range(num_classes):
        for k in range(N_GRID):
            ax = axes[c][k]
            ax.axis("off")
            if k < len(idxs[c]):
                img, _ = dataset[idxs[c][k]]
                it = img.unsqueeze(0).to(device)
                attr = lime_mnist_attr(model, it, c)
                vmax = max(np.max(np.abs(attr)), 1e-5)
                ax.imshow(denorm_mnist(it), cmap="gray")
                ax.imshow(attr, cmap="bwr", vmin=-vmax, vmax=vmax, alpha=0.6)
            if k == 0:
                ax.set_title(f"klasa {c}", loc="left", fontsize=9)
    fig.suptitle("MNIST CNN — LIME (superpiksele SLIC) dla wielu przykładów każdej klasy", fontsize=12)
    save_fig(fig, "mnist_cnn_lime_grid_all.png")


# ----------------------------------------------------------------------------
# MNIST MLP zoning16 — Feature Ablation 4x4
# ----------------------------------------------------------------------------
def mnist_mlp_zoning(model, test_ds, raw_imgs, device):
    print("[MNIST MLP zoning16] Feature Ablation 4x4 (siatka + uśrednienie)")
    num_classes = 10
    ablator = FeatureAblation(model)
    labels = np.array([int(test_ds[i][1]) for i in range(len(test_ds))])

    # zbierz indeksy per klasa
    idxs = {c: list(np.where(labels == c)[0][:N_MEAN]) for c in range(num_classes)}

    def attr_2d(gi, c):
        x = test_ds[gi][0].unsqueeze(0).to(device)
        a = ablator.attribute(x, target=c).squeeze(0).cpu().detach().numpy()
        return a.reshape(4, 4)

    # --- Siatka: oryginał + mapa 4x4 dla N_GRID przykładów, wszystkie klasy ---
    cols = N_GRID * 2  # para (oryginał, mapa)
    fig, axes = plt.subplots(num_classes, cols, figsize=(cols * 1.1, num_classes * 1.3))
    for c in range(num_classes):
        for k in range(N_GRID):
            ax_o = axes[c][2 * k]
            ax_a = axes[c][2 * k + 1]
            ax_o.axis("off")
            ax_a.axis("off")
            if k < len(idxs[c]):
                gi = idxs[c][k]
                ax_o.imshow(raw_imgs[gi], cmap="gray")
                a2d = attr_2d(gi, c)
                vmax = max(np.max(np.abs(a2d)), 1e-5)
                ax_a.imshow(a2d, cmap="bwr", vmin=-vmax, vmax=vmax, interpolation="nearest")
            if k == 0:
                ax_o.set_title(f"klasa {c}", loc="left", fontsize=9)
    fig.suptitle("MNIST MLP (zoning 4x4) — Feature Ablation dla wielu przykładów każdej klasy", fontsize=12)
    save_fig(fig, "mnist_mlp_zoning_grid_all.png")

    # --- Uśrednione mapy 4x4: 2x5 ---
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for c in range(num_classes):
        ax = axes[c // 5][c % 5]
        ax.axis("off")
        acc = np.zeros((4, 4))
        for gi in idxs[c]:
            acc += attr_2d(gi, c)
        acc /= max(len(idxs[c]), 1)
        vmax = max(np.max(np.abs(acc)), 1e-5)
        im = ax.imshow(acc, cmap="bwr", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.6)
        ax.set_title(f"cyfra {c}  (n={len(idxs[c])})", fontsize=9)
    fig.suptitle("MNIST MLP (zoning 4x4) — uśredniona atrybucja stref per klasa", fontsize=12)
    save_fig(fig, "mnist_mlp_zoning_mean_all.png")


# ----------------------------------------------------------------------------
# Imagenette — LIME (siatka per klasa) + Saliency uśredniony
# ----------------------------------------------------------------------------
def lime_imagenette_attr(model, image_tensor, target):
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    segments = slic(image_np, **SLIC_IMAGENETTE)
    fmask = torch.tensor(segments).unsqueeze(0).unsqueeze(0).to(image_tensor.device)
    baselines = image_tensor * 0.0
    lime = Lime(model)
    attr = lime.attribute(image_tensor, target=target, baselines=baselines,
                          feature_mask=fmask, n_samples=LIME_SAMPLES)
    a = attr.squeeze(0).cpu().detach().numpy()
    return np.sum(a, axis=0)


def imagenette_global(model, dataset, device):
    print("[Imagenette CNN] LIME per-klasa (siatki) + Saliency uśredniony")
    num_classes = 10
    # mniej skanowania: imagenette val ~3925, ale rozłożone -> przeskanuj wszystko
    idx_lime = find_correct_per_class(model, dataset, device, num_classes, N_GRID, batch_size=64)
    idx_mean = find_correct_per_class(model, dataset, device, num_classes,
                                      min(N_MEAN, 40), batch_size=64)

    # --- Siatki LIME: jedna figura na klasę (oryginał + nakładka LIME) ---
    for c in range(num_classes):
        if not idx_lime[c]:
            continue
        cols = len(idx_lime[c])
        fig, axes = plt.subplots(2, cols, figsize=(cols * 2.4, 5))
        if cols == 1:
            axes = axes.reshape(2, 1)
        for k, gi in enumerate(idx_lime[c]):
            img, _ = dataset[gi]
            it = img.unsqueeze(0).to(device)
            attr = lime_imagenette_attr(model, it, c)
            vmax = max(np.max(np.abs(attr)), 1e-9)
            disp = denorm_imagenette(it)
            axes[0][k].imshow(disp)
            axes[0][k].axis("off")
            axes[0][k].set_title(f"#{gi}", fontsize=8)
            axes[1][k].imshow(disp)
            axes[1][k].imshow(attr, cmap="bwr", vmin=-vmax, vmax=vmax, alpha=0.6)
            axes[1][k].axis("off")
        fig.suptitle(f"Imagenette — LIME, klasa {c} ({IMAGENETTE_CLASSES[c]})", fontsize=12)
        save_fig(fig, f"imagenette_lime_grid_class{c}.png")

    # --- Uśredniony Saliency per klasa: 2x5 ---
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
    for c in range(num_classes):
        ax = axes[c // 5][c % 5]
        ax.axis("off")
        if idx_mean[c]:
            acc = None
            for gi in idx_mean[c]:
                img, _ = dataset[gi]
                sal = saliency_map(model, img.unsqueeze(0).to(device), c)
                if sal.ndim == 3:
                    sal = sal.max(axis=0)
                acc = sal if acc is None else acc + sal
            acc /= len(idx_mean[c])
            ax.imshow(acc, cmap="hot")
        ax.set_title(f"{c}: {IMAGENETTE_CLASSES[c]}\n(n={len(idx_mean[c])})", fontsize=8)
    fig.suptitle("Imagenette CNN — uśredniona mapa Saliency per klasa", fontsize=12)
    save_fig(fig, "imagenette_saliency_mean_all.png")


# ----------------------------------------------------------------------------
def main():
    global N_GRID, N_MEAN, LIME_SAMPLES
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    if smoke:
        N_GRID, N_MEAN, LIME_SAMPLES = 2, 4, 40
        print(">>> TRYB SMOKE (mało próbek)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")
    t0 = time.time()

    # ---- MNIST CNN ----
    cnn = MnistCnnStandard()
    cnn.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_cnn_mnist.pt"), map_location=device))
    cnn.to(device)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
    mnist_test = torchvision.datasets.MNIST(root=os.path.join(PROJECT_ROOT, "data"),
                                            train=False, transform=tfm, download=True)
    mnist_cnn_saliency(cnn, mnist_test, device)
    mnist_cnn_lime(cnn, mnist_test, device)

    # ---- MNIST MLP zoning16 ----
    _, test_mlp, _, _, _ = load_mnist("zoning16")
    mlp = MLP(input_size=16, hidden_size=32, num_classes=10)
    mlp.load_state_dict(torch.load(os.path.join(MODELS_DIR, "MLP_MNIST_zoning16.pt"), map_location=device))
    mlp.to(device)
    raw_imgs = [mnist_test[i][0].squeeze().numpy() * MNIST_STD + MNIST_MEAN for i in range(len(test_mlp))]
    mnist_mlp_zoning(mlp, test_mlp, raw_imgs, device)

    # ---- Imagenette ----
    img_model = ImagenetteCnnStandard()
    img_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_cnn_imagenette.pt"), map_location=device))
    img_model.to(device)
    imagenette_test = get_dataset("imagenette", train=False, aug_type="none")
    imagenette_global(img_model, imagenette_test, device)

    print(f"\nGotowe! Czas: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
