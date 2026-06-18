import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision
import torchvision.transforms as transforms
from cnn_models import MnistCnnStandard
from explain_mnist import perturb_from_bottom

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(PROJECT_ROOT, 'wyniki_xai', 'bulk_perturbations')
    os.makedirs(out_dir, exist_ok=True)

    print("--- MNIST CNN Bulk Perturbations ---")
    model_cnn = MnistCnnStandard()
    model_cnn.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'models', 'best_cnn_mnist.pt'), map_location=device))
    model_cnn.to(device)
    
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_dir = os.path.join(PROJECT_ROOT, "data")
    mnist_test = torchvision.datasets.MNIST(root=data_dir, train=False, transform=tfm, download=True)
    
    def is_good_perturbation(model, image_tensor, original_class):
        img_clone = image_tensor.clone().detach()
        height = img_clone.shape[-2]
        for row in range(height-1, 10, -1): # zatrzymujemy się przed zasłonięciem całego (wiersz 10)
            img_clone[..., row, :] = -0.4242
            output = model(img_clone)
            pred_class = output.argmax(dim=1).item()
            if pred_class != original_class:
                return True
        return False

    good_indices = []
    # Przeszukujemy zbiór by znaleźć 30 przypadków
    for i in range(2000):
        sample_img, label = mnist_test[i]
        image_tensor = sample_img.unsqueeze(0).to(device)
        if is_good_perturbation(model_cnn, image_tensor, label):
            good_indices.append(i)
        if len(good_indices) >= 30:
            break

    print(f"Znaleziono {len(good_indices)} ciekawych przypadków. Generowanie obrazów...")
    for i in good_indices:
        sample_img, label = mnist_test[i]
        image_tensor = sample_img.unsqueeze(0).to(device)
        out_pert = os.path.join(out_dir, f'mnist_perturbation_sample{i}_class{label}.png')
        perturb_from_bottom(model_cnn, image_tensor, original_class=label, save_path=out_pert)

    print(f"Gotowe! Wygenerowano pliki w: {out_dir}")
