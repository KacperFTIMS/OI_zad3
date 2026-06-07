import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistCnnStandard(nn.Module):
    """Architektura 1 dla MNIST: Kilka warstw splotowych do standardowej klasyfikacji."""
    def __init__(self):
        super(MnistCnnStandard, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.AvgPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MnistCnn2D(nn.Module):
    """Architektura 2 dla MNIST: Kompresuje cechy do wektora 2D przed klasyfikacją."""
    def __init__(self):
        super(MnistCnn2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.AvgPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2d = nn.Linear(128, 2) 
        self.fc_out = nn.Linear(2, 10)

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        features_2d = self.fc2d(x)
        out = self.fc_out(features_2d)
        
        if return_features:
            return out, features_2d
        return out

class ImagenetteCnnStandard(nn.Module):
    """Architektura 1 dla Imagenette: Obrazy 3-kanałowe, wyższa rozdzielczość (160x160)."""
    def __init__(self):
        super(ImagenetteCnnStandard, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.AvgPool2d(2, 2)
        
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        
        self.fc1 = nn.Linear(128 * 10 * 10, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ImagenetteCnn2D(nn.Module):
    """Architektura 2 dla Imagenette: Podobna do Standard, ale redukuje cechy do 2D."""
    def __init__(self):
        super(ImagenetteCnn2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.AvgPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * 10 * 10, 512)
        self.fc2d = nn.Linear(512, 2)
        self.fc_out = nn.Linear(2, 10)

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        features_2d = self.fc2d(x)
        out = self.fc_out(features_2d)
        
        if return_features:
            return out, features_2d
        return out






class MnistCnnStandardV2(nn.Module):
    """Architektura 3 dla MNIST (Osoba 2): 3 warstwy splotowe, MaxPool2d, BatchNorm."""
    def __init__(self):
        super(MnistCnnStandardV2, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MnistCnn2DV2(nn.Module):
    """Architektura 4 dla MNIST (Osoba 2): 3 warstwy splotowe, MaxPool2d z wąskim gardłem 2D."""
    def __init__(self):
        super(MnistCnn2DV2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2d = nn.Linear(128, 2)
        self.fc_out = nn.Linear(2, 10)

    def forward(self, x, return_features=False):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        features_2d = self.fc2d(x)
        out = self.fc_out(features_2d)
        
        if return_features:
            return out, features_2d
        return out


class ImagenetteCnnStandardV2(nn.Module):
    """Architektura 3 dla Imagenette (Osoba 2): Klasyczna, 4 głębokie warstwy, Dropout2d, MaxPool2d."""
    def __init__(self):
        super(ImagenetteCnnStandardV2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2) 
        self.drop2d_1 = nn.Dropout2d(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2d_2 = nn.Dropout2d(0.1)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop2d_3 = nn.Dropout2d(0.2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop2d_4 = nn.Dropout2d(0.2)
        
        
        self.fc1 = nn.Linear(256 * 10 * 10, 512)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.drop2d_1(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2d_2(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop2d_3(self.pool(F.relu(self.bn3(self.conv3(x)))))
        x = self.drop2d_4(self.pool(F.relu(self.bn4(self.conv4(x)))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ImagenetteCnn2DV2(nn.Module):
    """Architektura 4 dla Imagenette (Osoba 2): Redukcja cech do 2D dla wykresów granicy decyzyjnej."""
    def __init__(self):
        super(ImagenetteCnn2DV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        
        self.fc1 = nn.Linear(256 * 10 * 10, 256)
        self.fc2d = nn.Linear(256, 2)
        self.fc_out = nn.Linear(2, 10)

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        features_2d = self.fc2d(x)
        out = self.fc_out(features_2d)
        
        if return_features:
            return out, features_2d
        return out
