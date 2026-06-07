import torch
from torchvision import datasets, transforms
import torchvision.transforms.v2 as v2
from torch.utils.data import Subset
import numpy as np

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, desc="Ładowanie zbioru do RAM"):
        print(f"{desc}...")
        self.data = []
        for i in range(len(dataset)):
            self.data.append(dataset[i])
            if (i + 1) % 1000 == 0:
                print(f"  Załadowano {i + 1}/{len(dataset)} próbek...")
        print(f"Gotowe! Załadowano {len(dataset)} próbek do RAM.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_transforms(dataset_name):
    """Zwraca bazowe transformacje dla zbioru (konwersja na tensor + ew. resize)."""
    if dataset_name == 'mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == 'imagenette':
        return transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    raise ValueError("Nieznany zbiór danych")

def get_gpu_transforms(dataset_name, aug_type='none'):
    """Zwraca transformacje z torchvision.transforms.v2 do nakładania na batche na GPU."""
    if aug_type == 'none':
        return torch.nn.Identity()

    if dataset_name == 'mnist':
        if aug_type == 'aug1':
            return v2.Compose([
                v2.RandomRotation(15)
            ])
        elif aug_type == 'aug2':
            return v2.Compose([
                v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ])
            
    elif dataset_name == 'imagenette':
        if aug_type == 'aug1':
            return v2.Compose([
                v2.Resize((180, 180)),
                v2.RandomCrop(160),
                v2.RandomHorizontalFlip()
            ])
        elif aug_type == 'aug2':
            return v2.Compose([
                v2.RandomResizedCrop(160, scale=(0.8, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ])
    raise ValueError("Nieznany zbiór danych lub typ augmentacji")


def get_dataset(dataset_name, train, aug_type='none'):
    
    transform = get_transforms(dataset_name)

    if dataset_name == 'mnist':
        ds = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'imagenette':
        split = 'train' if train else 'val'
        ds = datasets.Imagenette(root='./data', split=split, download=True, transform=transform)
    else:
        raise ValueError("Nieznany zbiór danych")
    
    return ds

def get_subset(dataset, num_samples):
    """
    Zwraca Subset datasetu składający się z num_samples próbek, 
    równomiernie rozłożonych pomiędzy wszystkie klasy.
    num_samples oznacza CAŁKOWITĄ liczbę próbek w nowym zbiorze.
    """
    if num_samples == 'all':
        return dataset
        
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, '_labels'):
        targets = np.array(dataset._labels)
    elif hasattr(dataset, '_samples'):
        targets = np.array([s[1] for s in dataset._samples])
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        
    classes = np.unique(targets)
    samples_per_class = num_samples // len(classes)
    
    indices = []
    for c in classes:
        c_indices = np.where(targets == c)[0]
        
        chosen_c_indices = np.random.choice(c_indices, samples_per_class, replace=False)
        indices.extend(chosen_c_indices)
        
    return Subset(dataset, indices)
