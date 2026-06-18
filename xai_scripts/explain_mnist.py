import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import Saliency, Lime
from skimage.segmentation import slic

def explain_with_saliency(model, image_tensor, target_class_idx, save_path):
    """
    Generuje mapę Saliency (wpływ pojedynczych pikseli) dla obrazu (np. MNIST).
    Pokazuje, gdzie model patrzył najbardziej przy podejmowaniu decyzji.
    """
    model.eval()
    
    saliency = Saliency(model)
    image_tensor.requires_grad_()
    
    attributions = saliency.attribute(image_tensor, target=target_class_idx)
    
    # Pobieramy same wartości absolutne (by widzieć po prostu obszary ważności, bez rozróżnienia na znak gradientu)
    attributions_np = np.abs(attributions.squeeze().cpu().detach().numpy())
    original_img = image_tensor.squeeze().cpu().detach().numpy()
    
    # Rysowanie wyników
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Oryginalny obraz')
    axes[0].axis('off')
    
    # Mapa Saliency w kolorach 'hot' dla lepszej wizualizacji (czerwony/żółty = ważne piksele)
    im = axes[1].imshow(attributions_np, cmap='hot')
    axes[1].set_title('Mapa Saliency (Ważność pikseli)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Zapisano Saliency do: {save_path}")
    plt.close()

def perturb_from_bottom(model, image_tensor, original_class, save_path):
    """
    Realizuje zadany w projekcie kontrprzykład, polegający na powolnym 
    zasłanianiu (czernieniu) cyfry od dołu wiersz po wierszu. Zatrzymuje się,
    gdy model zmieni klasyfikację cyfry na inną.
    """
    model.eval()
    img_clone = image_tensor.clone().detach()
    height = img_clone.shape[-2] # Wysokość obrazka, dla MNIST = 28
    
    for row in range(height-1, -1, -1):
        # Pobieramy oryginalne wartości wiersza, by móc je później wypisać
        original_row_values = img_clone[..., row, :].clone().flatten()
        
        # Zasłaniamy dany wiersz (kolor tła dla znormalizowanego MNIST to ok. -0.4242)
        img_clone[..., row, :] = -0.4242
        
        new_row_values = img_clone[..., row, :].flatten()
        
        # Predykcja na zmodyfikowanym obrazie
        output = model(img_clone)
        pred_class = output.argmax(dim=1).item()
        
        if pred_class != original_class:
            print(f"Predykcja uległa zmianie z klasy {original_class} na {pred_class} przy zasłonięciu do wiersza {row} (licząc od góry).")
            print(f"--- Szczegóły zmiany dla wiersza {row} ---")
            for i in range(len(original_row_values)):
                print(f"Piksel {i:2d}: Przed: {original_row_values[i]:.4f}, Po: {new_row_values[i]:.4f}")
            print("------------------------------------------")
            
            # Pokażmy wizualnie, w którym momencie model się pomylił
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            
            # Cofamy normalizację (wartości wracają do skali 0.0 - 1.0)
            orig_np = image_tensor.squeeze().cpu().detach().numpy() * 0.3081 + 0.1307
            clone_np = img_clone.squeeze().cpu().detach().numpy() * 0.3081 + 0.1307
            
            # Wymuszamy na matplotlibie sztywne widełki od 0 do 1
            axes[0].imshow(orig_np, cmap='gray', vmin=0.0, vmax=1.0)
            axes[0].set_title(f'Oryginał (Klasa {original_class})')
            axes[0].axis('off')
            
            axes[1].imshow(clone_np, cmap='gray', vmin=0.0, vmax=1.0)
            axes[1].set_title(f'Kontrprzykład ({pred_class})')
            axes[1].axis('off')
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Zapisano kontrprzykład do: {save_path}")
            plt.close()
            return
            
    print("Zasłonięto cały obraz od dołu do góry, a model cały czas zwracał poprawną klasę (lub po cichu zmienił w ostatniej iteracji).")

def explain_with_lime_mnist(model, image_tensor, target_class_idx, save_path):
    """
    Funkcja objaśniająca model na zbiorze MNIST przy użyciu LIME z segmentacją SLIC.
    """
    model.eval()
    image_np = image_tensor.squeeze(0).squeeze(0).cpu().detach().numpy()
    
    # Segmentacja SLIC dla obrazu 28x28 (odpowiednie parametry n_segments i compactness)
    segments = slic(image_np, n_segments=16, compactness=0.1, start_label=0, channel_axis=None)
    feature_mask = torch.tensor(segments).unsqueeze(0).unsqueeze(0).to(image_tensor.device)
    
    lime = Lime(model)
    # Tło w znormalizowanym MNIST to ok. -0.4242, więc używamy tej wartości jako baselines
    baselines = image_tensor * 0.0 - 0.4242 
    
    print("Trwa obliczanie atrybucji algorytmem LIME dla MNIST...")
    attributions = lime.attribute(
        image_tensor,
        target=target_class_idx,
        baselines=baselines,
        feature_mask=feature_mask,
        n_samples=200 
    )
    
    attr_np = attributions.squeeze(0).squeeze(0).cpu().detach().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Cofamy normalizację (wartości wracają do skali 0.0 - 1.0)
    orig_np = image_tensor.squeeze().cpu().detach().numpy() * 0.3081 + 0.1307
    
    axes[0].imshow(orig_np, cmap='gray', vmin=0.0, vmax=1.0)
    axes[0].set_title('Oryginał MNIST')
    axes[0].axis('off')
    
    vmax = np.max(np.abs(attr_np))
    # Jeżeli vmax jest zbyt bliskie 0 (brak wpływu), ustalamy minimalne vmax
    if vmax < 1e-5:
        vmax = 1e-5
        
    im = axes[1].imshow(attr_np, cmap='bwr', vmin=-vmax, vmax=vmax) 
    axes[1].set_title('Wpływ superpikseli (LIME)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Zapisano LIME MNIST do: {save_path}")
    plt.close()

def explain_with_ablation_zoning2d(model, input_tensor, original_img_2d, target_class_idx, save_path):
    """
    Generuje dwuwymiarowy (4x4) wykres atrybucji Feature Ablation dla modelu MLP (zoning16).
    Pokazuje rozkład ważności cech w przestrzeni 2D.
    """
    model.eval()
    
    # Inicjalizacja Feature Ablation
    from captum.attr import FeatureAblation
    ablator = FeatureAblation(model)
    attributions = ablator.attribute(input_tensor, target=target_class_idx)
    attributions_np = attributions.squeeze(0).cpu().detach().numpy()
    
    # Reshape do siatki 4x4
    attr_2d = attributions_np.reshape(4, 4)
    
    # Rysowanie
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Oryginalny obrazek
    axes[0].imshow(original_img_2d, cmap='gray')
    axes[0].set_title(f'Oryginalny obraz (Klasa {target_class_idx})')
    axes[0].axis('off')
    
    # Wizualizacja 2D atrybucji (siatka 4x4)
    vmax = np.max(np.abs(attr_2d))
    if vmax < 1e-5:
        vmax = 1e-5
        
    im = axes[1].imshow(attr_2d, cmap='bwr', vmin=-vmax, vmax=vmax, interpolation='nearest')
    axes[1].set_title('Wpływ stref 4x4 (Ablation)')
    axes[1].axis('off')
    
    # Dodanie siatki pomocniczej do obrazka po prawej
    axes[1].set_xticks(np.arange(-0.5, 4, 1), minor=True)
    axes[1].set_yticks(np.arange(-0.5, 4, 1), minor=True)
    axes[1].grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Zapisano wizualizację 2D MLP do: {save_path}")
    plt.close()

# --- MIEJSCE NA TWOJE KODY ---
if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import torchvision
    import torchvision.transforms as transforms
    from cnn_models import MnistCnnStandard
    from yep import MLP
    from dataloader import load_mnist
    from explain_tabular import explain_tabular_model

    print("--- 1. Objaśnianie modelu CNN dla zbioru MNIST ---")
    model_cnn = MnistCnnStandard()
    cnn_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_cnn_mnist.pt')
    model_cnn.load_state_dict(torch.load(cnn_path, map_location='cpu'))
    
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_test = torchvision.datasets.MNIST(root="data", train=False, transform=tfm, download=True)
    sample_img, label = mnist_test[0]
    image_tensor = sample_img.unsqueeze(0)

    out_saliency = os.path.join(os.path.dirname(__file__), '..', 'wyniki_xai', 'mnist_saliency.png')
    out_perturb = os.path.join(os.path.dirname(__file__), '..', 'wyniki_xai', 'mnist_perturbation.png')
    explain_with_saliency(model_cnn, image_tensor, target_class_idx=label, save_path=out_saliency)
    perturb_from_bottom(model_cnn, image_tensor, original_class=label, save_path=out_perturb)

    print("--- 2. Objaśnianie modelu MLP (zoning16) dla zbioru MNIST ---")
    _, test_dataset_mlp, input_size, num_classes, _ = load_mnist('zoning16')
    model_mlp = MLP(input_size=16, hidden_size=32, num_classes=10)
    mlp_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'MLP_MNIST_zoning16.pt')
    model_mlp.load_state_dict(torch.load(mlp_path, map_location='cpu'))

    # Pobieramy próbkę z przetworzonego już test_dataset
    sample_mlp_tensor, label_mlp_tensor = test_dataset_mlp[0]
    # Dataset zwraca FloatTensor bez wymiaru batcha, więc robimy unsqueeze
    sample_mlp_input = sample_mlp_tensor.unsqueeze(0)
    label_mlp = label_mlp_tensor.item()
    
    zoning_feature_names = [f"Strefa {i+1}" for i in range(16)]
    out_zoning = os.path.join(os.path.dirname(__file__), '..', 'wyniki_xai', 'mnist_mlp_zoning_ablation.png')
    explain_tabular_model(model_mlp, sample_mlp_input, target_class_idx=label_mlp, feature_names=zoning_feature_names, save_path=out_zoning)

    # Generowanie wizualizacji 2D
    original_img_2d = mnist_test[0][0].squeeze().numpy() * 0.3081 + 0.1307
    out_zoning_2d = os.path.join(os.path.dirname(__file__), '..', 'wyniki_xai', 'mnist_mlp_zoning_2d.png')
    explain_with_ablation_zoning2d(model_mlp, sample_mlp_input, original_img_2d, label_mlp, out_zoning_2d)

