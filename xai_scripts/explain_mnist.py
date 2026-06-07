import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import Saliency

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
        # Zasłaniamy dany wiersz (ustawiamy wartość pikseli na 0 - kolor tła)
        img_clone[..., row, :] = 0.0
        
        # Predykcja na zmodyfikowanym obrazie
        output = model(img_clone)
        pred_class = output.argmax(dim=1).item()
        
        if pred_class != original_class:
            print(f"Predykcja uległa zmianie z klasy {original_class} na {pred_class} przy zasłonięciu do wiersza {row} (licząc od góry).")
            
            # Pokażmy wizualnie, w którym momencie model się pomylił
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(image_tensor.squeeze().cpu().detach().numpy(), cmap='gray')
            axes[0].set_title(f'Oryginał (Klasa {original_class})')
            axes[0].axis('off')
            
            axes[1].imshow(img_clone.squeeze().cpu().detach().numpy(), cmap='gray')
            axes[1].set_title(f'Kontrprzykład (Pomylone na {pred_class})')
            axes[1].axis('off')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Zapisano kontrprzykład do: {save_path}")
            plt.close()
            return
            
    print("Zasłonięto cały obraz od dołu do góry, a model cały czas zwracał poprawną klasę (lub po cichu zmienił w ostatniej iteracji).")

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
    
    tfm = transforms.Compose([transforms.ToTensor()])
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

