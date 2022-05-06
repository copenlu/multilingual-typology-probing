from typing import Optional, List
from overrides import overrides
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import numpy as np

from probekit.utils.types import PyTorchDevice, Specification
from probekit.utils.dataset import FastTensorDataLoader, ClassificationDataset
from probekit.trainers.neural_probe_trainer import NeuralProbeTrainer
from probekit.models.discriminative.neural_probe import NeuralProbeModel


class VariationalFamily(nn.Module):
    def __init__(self, D: int):
        super().__init__()

        self.D = D
        self.weights = nn.Parameter(torch.rand(self.D))

    def entropy(self):
        # other marginals
        marginals = self.weights / (1 + self.weights)
        logs = -torch.log(self.weights)
        return torch.dot(marginals, logs) + self.logZ()

    def logZ(self):
        # linear-time DP in log-space
        logZ = 0.0
        for d in range(self.D):
            logZ += torch.log1p(self.weights[d])

        return logZ


D = 100
vf = VariationalFamily(D)
print(vf.logZ())
x = vf.entropy()
