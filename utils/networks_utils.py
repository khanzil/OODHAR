import torch
import torch.nn as nn
import torchvision 
import torch.nn.functional as F
from torch.autograd import Function


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, cfg):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, cfg['mlp_width'])
        self.dropout = nn.Dropout(cfg['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(cfg['mlp_width'], cfg['mlp_width'])
            for _ in range(cfg['mlp_depth']-2)])
        self.output = nn.Linear(cfg['mlp_width'], n_outputs)
        self.n_outputs = n_outputs
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        x = self.activation(x) # for URM; does not affect other algorithms
        return x

class ResNet(nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, cfg):
        super(ResNet, self).__init__()
        if cfg['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = nn.Identity()

        if cfg["freeze_bn"]:
            self.freeze_bn()
        self.cfg = cfg
        self.dropout = nn.Dropout(cfg['resnet_dropout'])
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.activation(self.dropout(self.network(x)))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.cfg["freeze_bn"]:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
  
class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return self.activation(x)

class GradReverse(Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return -grad_output

class GRL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return GradReverse.apply(x)

featurizer_dict = {
    'MLP'       : MLP,
    'MNIST_CNN' : MNIST_CNN,
    'ResNet'    : ResNet
}

# This include the backbone networks
def Featurizer(input_shape, cfg):
    """
        Select feature extractor    
    """
    
    match cfg["model"]["featurizer"]:
        case "MLP":
            if len(input_shape) == 1:
                raise ValueError(f"Expect input_shape for MLP model to have len 1, get {len(input_shape)}")
            return MLP(input_shape[0], cfg["model"]["mlp_width"], cfg["model"])
        case "MNIST_CNN":
            return MNIST_CNN(input_shape)
        case "ResNet":
            return ResNet(input_shape, cfg["model"])
        case _:
            raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)



























