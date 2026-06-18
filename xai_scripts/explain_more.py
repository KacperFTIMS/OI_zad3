import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision
import torchvision.transforms as transforms
from cnn_models import MnistCnnStandard, ImagenetteCnnStandard
from cnn_data import get_dataset
from dataloader import load_mnist
from yep import MLP
from sklearn.datasets import load_iris, load_wine

# Import explanation functions from existing scripts
from explain_tabular import explain_tabular_model
from explain_mnist import explain_with_saliency, perturb_from_bottom, explain_with_lime_mnist
from explain_imagenette import explain_with_lime_slic

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(PROJECT_ROOT, 'wyniki_xai')
    os.makedirs(out_dir, exist_ok=True)

    # 1. Iris (3 samples, 1 per class)
    print("--- Iris ---")
    iris = load_iris()
    model_iris = MLP(input_size=4, hidden_size=16, num_classes=3)
    model_iris.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'models', 'MLP_Iris.pt'), map_location=device))
    model_iris.to(device)
    # Classes are grouped in load_iris (0-49: class 0, 50-99: class 1, 100-149: class 2)
    for idx in [0, 50, 100]:
        sample = torch.tensor(iris.data[idx:idx+1], dtype=torch.float32).to(device)
        label = int(iris.target[idx])
        out_path = os.path.join(out_dir, f'iris_ablation_class{label}.png')
        explain_tabular_model(model_iris, sample, target_class_idx=label, feature_names=iris.feature_names, save_path=out_path)

    # 2. Wine (3 samples, 1 per class)
    print("--- Wine ---")
    wine = load_wine()
    model_wine = MLP(input_size=13, hidden_size=26, num_classes=3)
    model_wine.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'models', 'MLP_Wine.pt'), map_location=device))
    model_wine.to(device)
    # Classes in load_wine: 0-58: 0, 59-129: 1, 130-177: 2
    for idx in [0, 60, 131]:
        sample = torch.tensor(wine.data[idx:idx+1], dtype=torch.float32).to(device)
        label = int(wine.target[idx])
        out_path = os.path.join(out_dir, f'wine_ablation_class{label}.png')
        explain_tabular_model(model_wine, sample, target_class_idx=label, feature_names=wine.feature_names, save_path=out_path)

    # 3. MNIST CNN (tylko próbki wybrane do raportu)
    print("--- MNIST CNN ---")
    model_cnn = MnistCnnStandard()
    model_cnn.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'models', 'best_cnn_mnist.pt'), map_location=device))
    model_cnn.to(device)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_dir = os.path.join(PROJECT_ROOT, "data")
    mnist_test = torchvision.datasets.MNIST(root=data_dir, train=False, transform=tfm, download=True)
    
    # Próbki zdefiniowane w raporcie: Saliency dla indeksu 30, Perturbacje dla indeksów 30 i 33, LIME dla 11, 15, 30, 33.
    for i in [11, 15, 30, 33]:
        sample_img, label = mnist_test[i]
        image_tensor = sample_img.unsqueeze(0).to(device)
        out_pert = os.path.join(out_dir, f'mnist_perturbation_sample{i}_class{label}.png')
        perturb_from_bottom(model_cnn, image_tensor, original_class=label, save_path=out_pert)
        
        # Saliency generujemy tylko dla 30 (cyfra 3) aby nie zaśmiecać
        if i == 30:
            out_sal = os.path.join(out_dir, f'mnist_saliency_sample{i}_class{label}.png')
            explain_with_saliency(model_cnn, image_tensor, target_class_idx=label, save_path=out_sal)
            
        out_lime = os.path.join(out_dir, f'mnist_lime_sample{i}_class{label}.png')
        explain_with_lime_mnist(model_cnn, image_tensor, target_class_idx=label, save_path=out_lime)

    # 4. Imagenette CNN (3 different samples)
    print("--- Imagenette CNN ---")
    model_img = ImagenetteCnnStandard()
    model_img.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'models', 'best_cnn_imagenette.pt'), map_location=device))
    model_img.to(device)
    imagenette_test = get_dataset('imagenette', train=False, aug_type='none')
    # Pick a few samples (including other classes: 1 - English springer, 4 - church, 8 - golf ball)
    for i in [10, 20, 40, 90, 387, 1525, 3136]:
        sample_img, label = imagenette_test[i]
        image_tensor = sample_img.unsqueeze(0).to(device)
        out_path = os.path.join(out_dir, f'imagenette_lime_sample{i}_class{label}.png')
        explain_with_lime_slic(model_img, image_tensor, target_class_idx=label, save_path=out_path)

    print("Gotowe!")
