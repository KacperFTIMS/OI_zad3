import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yep import MLP
from dataloader import load_data, load_mnist
from cnn_data import get_dataset
from cnn_models import MnistCnnStandard, ImagenetteCnnStandard

def evaluate(model, test_loader, device='cpu'):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100.0 * correct / total

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. MLP Iris
    print("\nEvaluating MLP Iris...")
    _, test_iris, _, _ = load_data('iris')
    model_iris = MLP(input_size=4, hidden_size=16, num_classes=3)
    model_iris.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'models', 'MLP_Iris.pt'), map_location=device))
    loader_iris = DataLoader(test_iris, batch_size=32, shuffle=False)
    acc_iris = evaluate(model_iris, loader_iris, device)
    print(f"MLP Iris Accuracy: {acc_iris:.2f}%")
    
    # 2. MLP Wine
    print("\nEvaluating MLP Wine...")
    _, test_wine, _, _ = load_data('wine')
    model_wine = MLP(input_size=13, hidden_size=26, num_classes=3)
    model_wine.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'models', 'MLP_Wine.pt'), map_location=device))
    loader_wine = DataLoader(test_wine, batch_size=32, shuffle=False)
    acc_wine = evaluate(model_wine, loader_wine, device)
    print(f"MLP Wine Accuracy: {acc_wine:.2f}%")
    
    # 3. MLP MNIST (zoning16)
    print("\nEvaluating MLP MNIST (zoning16)...")
    _, test_zoning, _, _, _ = load_mnist('zoning16')
    model_zoning = MLP(input_size=16, hidden_size=32, num_classes=10)
    model_zoning.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'models', 'MLP_MNIST_zoning16.pt'), map_location=device))
    loader_zoning = DataLoader(test_zoning, batch_size=256, shuffle=False)
    acc_zoning = evaluate(model_zoning, loader_zoning, device)
    print(f"MLP MNIST (zoning16) Accuracy: {acc_zoning:.2f}%")
    
    # 4. CNN MNIST
    print("\nEvaluating CNN MNIST...")
    test_cnn_mnist = get_dataset('mnist', train=False)
    model_cnn_mnist = MnistCnnStandard()
    model_cnn_mnist.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'models', 'best_cnn_mnist.pt'), map_location=device))
    loader_cnn_mnist = DataLoader(test_cnn_mnist, batch_size=256, shuffle=False)
    acc_cnn_mnist = evaluate(model_cnn_mnist, loader_cnn_mnist, device)
    print(f"CNN MNIST Accuracy: {acc_cnn_mnist:.2f}%")
    
    # 5. CNN Imagenette
    print("\nEvaluating CNN Imagenette...")
    test_cnn_img = get_dataset('imagenette', train=False)
    model_cnn_img = ImagenetteCnnStandard()
    model_cnn_img.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'models', 'best_cnn_imagenette.pt'), map_location=device))
    loader_cnn_img = DataLoader(test_cnn_img, batch_size=64, shuffle=False)
    acc_cnn_img = evaluate(model_cnn_img, loader_cnn_img, device)
    print(f"CNN Imagenette Accuracy: {acc_cnn_img:.2f}%")
