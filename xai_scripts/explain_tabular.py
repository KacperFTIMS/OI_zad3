import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import FeatureAblation

def explain_tabular_model(model, input_tensor, target_class_idx, feature_names, save_path):
    """
    Funkcja objaśniająca model tabelaryczny (np. dla Iris lub Wine) 
    z wykorzystaniem metody Feature Ablation.
    """
    model.eval()
    
    # Inicjalizacja Feature Ablation
    ablator = FeatureAblation(model)
    
    # Obliczanie atrybucji (wpływu) poszczególnych cech
    # baseline to domyślnie 0, co oznacza wyzerowanie cechy na potrzeby sprawdzenia jak model zareaguje
    attributions = ablator.attribute(input_tensor, target=target_class_idx)
    
    # Konwersja do numpy dla łatwiejszego rysowania wykresu
    attributions_np = attributions.squeeze(0).cpu().detach().numpy()
    
    # Wypisywanie wyników w tabeli w konsoli
    print(f"\nTablica atrybucji dla klasy {target_class_idx}:")
    print(f"{'Cecha':<30} | {'Atrybucja':>10}")
    print("-" * 43)
    for name, attr_val in zip(feature_names, attributions_np):
        print(f"{name:<30} | {attr_val:>10.4f}")
    print("-" * 43)

    # Rysowanie wykresu słupkowego
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, attributions_np, align='center', color='skyblue', edgecolor='black')
    plt.yticks(y_pos, feature_names)
    plt.xlabel('Wartość atrybucji (Wpływ na predykcję)')
    plt.title(f'Feature Ablation - Ważność cech dla klasy {target_class_idx}')
    plt.axvline(x=0, color='black', linewidth=1)  # Linia zera
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Zapisano wykres do: {save_path}")
    plt.close()

# --- MIEJSCE NA TWOJE KODY ---
if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from yep import MLP
    from sklearn.datasets import load_iris, load_wine

    print("--- Objaśnianie modelu dla zbioru Iris ---")
    iris = load_iris()
    iris_features = iris.feature_names
    # Iris: 4 cechy wejściowe, 3 klasy
    # Zgodnie z kodem projektu: hidden_size = max(16, 4 * 2) = 16
    model_iris = MLP(input_size=4, hidden_size=16, num_classes=3)
    model_path_iris = os.path.join(os.path.dirname(__file__), '..', 'models', 'MLP_Iris.pt')
    model_iris.load_state_dict(torch.load(model_path_iris, map_location='cpu'))
    
    # Generujemy dla każdej z klas po jednym przykładzie
    for idx in [0, 50, 100]:
        sample_iris = torch.tensor(iris.data[idx:idx+1], dtype=torch.float32)
        label_iris = int(iris.target[idx])
        out_path_iris = os.path.join(os.path.dirname(__file__), '..', 'wyniki_xai', f'iris_ablation_class{label_iris}.png')
        explain_tabular_model(model_iris, sample_iris, target_class_idx=label_iris, feature_names=iris_features, save_path=out_path_iris)

    print("\n--- Objaśnianie modelu dla zbioru Wine ---")
    wine = load_wine()
    wine_features = wine.feature_names
    # Wine: 13 cech wejściowych, 3 klasy
    # Zgodnie z kodem projektu: hidden_size = max(16, 13 * 2) = 26
    model_wine = MLP(input_size=13, hidden_size=26, num_classes=3)
    model_path_wine = os.path.join(os.path.dirname(__file__), '..', 'models', 'MLP_Wine.pt')
    model_wine.load_state_dict(torch.load(model_path_wine, map_location='cpu'))
    
    # Generujemy dla każdej z klas po jednym przykładzie
    for idx in [0, 60, 131]:
        sample_wine = torch.tensor(wine.data[idx:idx+1], dtype=torch.float32)
        label_wine = int(wine.target[idx])
        out_path_wine = os.path.join(os.path.dirname(__file__), '..', 'wyniki_xai', f'wine_ablation_class{label_wine}.png')
        explain_tabular_model(model_wine, sample_wine, target_class_idx=label_wine, feature_names=wine_features, save_path=out_path_wine)
