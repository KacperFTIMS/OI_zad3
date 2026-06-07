import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import Lime
from skimage.segmentation import slic

def explain_with_lime_slic(model, image_tensor, target_class_idx, save_path):
    """
    Funkcja objaśniająca model obrazowy (np. CNN dla Imagenette) 
    z wykorzystaniem podejścia lokalnego LIME oraz segmentacji SLIC, 
    która dzieli obraz na tzw. "interpretowalne komponenty" (superpiksele).
    """
    model.eval()
    
    # ============================================================
    # 1. Segmentacja SLIC (tworzenie "superpikseli")
    # ============================================================
    # Imagenette to często tensory np. [1, 3, 224, 224] i mogą być znormalizowane.
    # W skimage musimy przenieść kanały na koniec dla formatu np. (224, 224, 3)
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    
    # Konfiguracja SLIC: n_segments określa docelową liczbę superpikseli, compactness ich spójność
    # Wartości te (szczególnie n_segments) warto modyfikować w ramach eksperymentów.
    segments = slic(image_np, n_segments=50, compactness=10, start_label=0)
    
    # Captum potrzebuje maski jako tensor kształtu [1, 1, 224, 224] (dla każdego piksela przypisane ID grupy)
    feature_mask = torch.tensor(segments).unsqueeze(0).unsqueeze(0).to(image_tensor.device)
    
    # ============================================================
    # 2. Inicjalizacja LIME i obliczanie atrybucji
    # ============================================================
    lime = Lime(model)
    
    # Tworzymy baseline - czyli to, czym ma być zasłonięty superpiksel. Najczęściej sprawdzamy czarne pole.
    baselines = image_tensor * 0.0
    
    print("Trwa obliczanie atrybucji algorytmem LIME. Może to potrwać dłuższą chwilę...")
    attributions = lime.attribute(
        image_tensor,
        target=target_class_idx,
        baselines=baselines,
        feature_mask=feature_mask,
        n_samples=200 # Liczba próbek generowanych do wytrenowania modelu liniowego (jeśli za mało, mogą być złe wyniki)
    )
    
    # ============================================================
    # 3. Wizualizacja
    # ============================================================
    attr_np = attributions.squeeze(0).cpu().detach().numpy()
    
    # Obraz atrybucji LIME też ma 3 kanały (C, H, W). Sumujemy je po wymiarze kanałów (0), 
    # żeby otrzymać prostą mapę cieplną (H, W).
    if len(attr_np.shape) == 3:
        attr_np = np.sum(attr_np, axis=0) 
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Jeśli image_np ma wartości ujemne przez transform.Normalize, tutaj tylko brutalnie ucinamy (clip), 
    # dla ładniejszego wyświetlania wypadałoby cofnąć normalizację.
    img_viz = np.clip(image_np, 0, 1)
    
    axes[0].imshow(img_viz)
    axes[0].set_title('Oryginalny obraz z Imagenette')
    axes[0].axis('off')
    
    # Heatmapa (niebieski - osłabia prawdopodobieństwo, czerwony - wzmacnia)
    vmax = np.max(np.abs(attr_np))
    im = axes[1].imshow(attr_np, cmap='bwr', vmin=-vmax, vmax=vmax) 
    axes[1].set_title('Wpływ superpikseli (LIME)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Zapisano LIME do: {save_path}")
    plt.close()

# --- MIEJSCE NA TWOJE KODY ---
if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from cnn_models import ImagenetteCnnStandard
    from cnn_data import get_dataset

    print("--- Objaśnianie modelu CNN dla zbioru Imagenette ---")
    model = ImagenetteCnnStandard()
    # map_location, aby załadowało się nawet gdy trenowano na GPU
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_cnn_imagenette.pt')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    imagenette_test = get_dataset('imagenette', train=False, aug_type='none')
    sample_img, label = imagenette_test[0]
    image_tensor = sample_img.unsqueeze(0)
    
    out_path = os.path.join(os.path.dirname(__file__), '..', 'wyniki_xai', 'imagenette_lime.png')
    explain_with_lime_slic(model, image_tensor, target_class_idx=label, save_path=out_path)
