from typing import Optional
from overrides import overrides
import torch

from probekit.utils.types import PyTorchDevice
from probekit.utils.dataset import ClassificationDataset
from probekit.models.discriminative.neural_probe import NeuralProbeModel

from .flexible_neural_probe_trainer import FlexibleNeuralProbeTrainer


class SimpleTrainer(FlexibleNeuralProbeTrainer):
    """
    Used to implement upperbound and lowerbound (Dalvi's) probe.
    """
    def __init__(
            self, model: NeuralProbeModel, dataset: ClassificationDataset, device: PyTorchDevice,
            decomposable: bool = True, lr: float = 1e-2, num_epochs: int = 2000,
            batch_size: Optional[int] = None, report_progress: bool = True,
            patience: int = 50, l1_weight: float = 0.0, l2_weight: float = 0.0):
        super().__init__(model=model, dataset=dataset, device=device, decomposable=decomposable, lr=lr,
                         num_epochs=num_epochs, batch_size=batch_size, report_progress=report_progress,
                         temp_annealing=False, patience=patience, l1_weight=l1_weight, l2_weight=l2_weight)

    @overrides
    def _sample_mask(self, shape, temperature: float = 0.0, device: PyTorchDevice = "cpu") -> torch.Tensor:
        return torch.ones(shape, device=device)
