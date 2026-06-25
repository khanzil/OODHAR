import torch
import torch.nn as nn
import torchvision 
import torch.nn.functional as F
from torch.autograd import Function
from collections import OrderedDict
from numbers import Number
import operator


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, cfgs):
        super(MLP, self).__init__()
        self.flat = nn.Flatten()
        self.input = nn.Linear(n_inputs, cfgs['mlp_width'])
        self.dropout = nn.Dropout(cfgs['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(cfgs['mlp_width'], cfgs['mlp_width'])
            for _ in range(cfgs['mlp_depth']-2)])
        self.output = nn.Linear(cfgs['mlp_width'], cfgs['mlp_num_hidden'])
        self.n_outputs = cfgs['mlp_num_hidden']
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.flat(x)
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
    def __init__(self, input_shape, cfgs):
        super(ResNet, self).__init__()
        if cfgs['resnet18']:
            if cfgs['pretrain']:
                self.network = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            else:
                self.network = torchvision.models.resnet18()
            self.n_outputs = 512
        else:
            if cfgs['pretrain']:
                self.network = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            else:
                self.network = torchvision.models.resnet50()
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

        if cfgs['freeze_bn']:
            self.freeze_bn()
        self.cfgs = cfgs
        self.dropout = nn.Dropout(cfgs['resnet_dropout'])
        self.activation = nn.Identity() # for URM; does not affect other algorithms
        # torchvision Resnet already have a flatten layer 

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.activation(self.dropout(self.network(x)))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.cfgs["freeze_bn"]:
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
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.save_for_backward(x, lambd)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, lambd = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - lambd*grad_output
        return grad_input, None

class GRL(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = torch.tensor(lambd, requires_grad=False)

    def forward(self,x):
        return GradReverse().apply(x,self.lambd)

featurizer_dict = {
    'MLP'       : MLP,
    'MNIST_CNN' : MNIST_CNN,
    'ResNet'    : ResNet
}

# This include the backbone networks
def Featurizer(cfgs):
    """
        Select feature extractor    
    """

    input_shape = [int(item.strip()) for item in cfgs['num_inputs'].split(',')]
    
    match cfgs["featurizer"]:
        case "MLP":
            return MLP(torch.sum(torch.ones(input_shape)).int(), cfgs)
        case "MNIST_CNN":
            return MNIST_CNN(input_shape)
        case "ResNet":
            return ResNet(input_shape, cfgs)
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


class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)

























