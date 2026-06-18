import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import torch
from yep import *

from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(dataset):
    if dataset == 'iris':
        data = load_iris()
    elif dataset == 'wine':
        data = load_wine()

    X = data.data
    y = data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = SwagDataset(X_train, y_train)
    test_dataset = SwagDataset(X_test, y_test)

    return train_dataset, test_dataset, X.shape[1], len(data.target_names)

def extract_flatten(img_tensor):
    return img_tensor.view(-1).numpy()

def extract_crossings2d(img_tensor):
    img = img_tensor.squeeze(0).numpy()
    bin_img = img > 0.1
    
    max_crossings_x = 0
    for row in bin_img:
        crossings = np.sum(row[1:] > row[:-1]) + int(row[0])
        if crossings > max_crossings_x:
            max_crossings_x = crossings
            
    max_crossings_y = 0
    for col in bin_img.T:
        crossings = np.sum(col[1:] > col[:-1]) + int(col[0])
        if crossings > max_crossings_y:
            max_crossings_y = crossings
            
    return np.array([float(max_crossings_x), float(max_crossings_y)])

def extract_projections(img_tensor): # 56 cech splaszczenie pionowo i poziomo
    img = img_tensor.squeeze(0).numpy()
    proj_v = np.sum(img, axis=0) 
    proj_h = np.sum(img, axis=1) 

    return np.concatenate([proj_v, proj_h])

def extract_symmetry2d(img_tensor): # 2 cechy symetria pionowa i pozioma
    img = img_tensor.squeeze(0).numpy()
    img_flipped_h = np.fliplr(img)
    img_flipped_v = np.flipud(img)
    sym_h = -np.sum(np.abs(img - img_flipped_h))
    sym_v = -np.sum(np.abs(img - img_flipped_v))
    
    return np.array([sym_h, sym_v])

def extract_zoning16(img_tensor):
    img = img_tensor.squeeze(0).numpy()
    blocks = img.reshape(4, 7, 4, 7)
    zoning_features = blocks.mean(axis=(1, 3))
    return zoning_features.flatten()

def load_mnist(extractor_name):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extractors = {
        'flatten': extract_flatten,
        'crossings2d': extract_crossings2d,
        'symmetry2d': extract_symmetry2d,
        'zoning16': extract_zoning16,
        'projections': extract_projections
    }
    extractor_fn = extractors[extractor_name]
    out_path = os.path.join(PROJECT_ROOT, f"mnist_{extractor_name}.npz")
    
    if not os.path.exists(out_path):
        print(f"Generowanie cech dla MNIST ({extractor_name})...")
        tfm = transforms.Compose([transforms.ToTensor()])
        data_dir = os.path.join(PROJECT_ROOT, "data")
        mnist_train = torchvision.datasets.MNIST(root=data_dir, train=True, transform=tfm, download=True)
        mnist_test = torchvision.datasets.MNIST(root=data_dir, train=False, transform=tfm, download=True)
        
        X_train_list, y_train_list = [], []
        for img, label in mnist_train:
            X_train_list.append(extractor_fn(img))
            y_train_list.append(label)
            
        X_test_list, y_test_list = [], []
        for img, label in mnist_test:
            X_test_list.append(extractor_fn(img))
            y_test_list.append(label)
            
        X_train = np.stack(X_train_list).astype(np.float32)
        y_train = np.array(y_train_list, dtype=np.int64)
        X_test = np.stack(X_test_list).astype(np.float32)
        y_test = np.array(y_test_list, dtype=np.int64)
        
        np.savez(out_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    
    data = np.load(out_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_dataset = SwagDataset(X_train, y_train)
    test_dataset = SwagDataset(X_test, y_test)
    
    return train_dataset, test_dataset, X_train.shape[1], 10, out_path
