from typing import Optional, List, Dict, Any, Union
import torch
import torch.nn as nn

import time

from .flexible_neural_probe_trainer import FlexibleNeuralProbeTrainer


class VariationalFamily(nn.Module):
    def __init__(self, D: int):
        super().__init__()

        self.D = D
        self.weights = nn.Parameter(torch.rand(self.D))

    def marginals(self):
        w = self.weights.exp()
        return w / (1 + w)

    def entropy(self):
        w = self.weights.exp()
        logs = -torch.log(w)
        return torch.dot(self.marginals(), logs) + self.logZ()

    def logZ(self):
        w = self.weights.exp()
        logZ = torch.sum(torch.log1p(w))
        return logZ

    def logprob(self, mask):
        w = self.weights.exp()
        logs = torch.log(w)
        logprob = mask.float() @ logs - self.logZ()
        return logprob

    def sample(self, N):
        return torch.rand(N, self.D, device=self.weights.device) < self.marginals()


class PoissonTrainer(FlexibleNeuralProbeTrainer):
    def __init__(self, mc_samples: int = 5, entropy_scale: float = 1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mc_samples = mc_samples
        self.variational_family = VariationalFamily(self._dataset.get_dimensionality()).to(self.get_device())
        self.entropy_scale = torch.tensor([entropy_scale], device=self.get_device())

    def setup_parameter_list(self):
        params_to_optimize = super().setup_parameter_list()
        params_to_optimize.extend(self.variational_family.parameters())

        return params_to_optimize

    def train_loss(self, minibatch: torch.Tensor, minibatch_assignment: torch.Tensor,
                   select_dimensions: Optional[List[int]], others: Dict[str, Any]):
        minibatch_size, dim = minibatch.shape
        num_values = len(self.get_property_values())
        model = self._model
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        # If select_dimensions is set, we want to train the probe statically on a specific set of dimensions
        # (as it isn't decomposable).
        if select_dimensions:
            # NOTE: This case shouldn't arise unless we try to train with manual mode.
            raise NotImplementedError("Need to think about how to handle this case, if it ever arises.")

        # COMPUTE FIRST TERM OF LOSS, AKA., THE MC ESTIMATE OF THE LIKELIHOOD UNDER Q_\VPHI
        # Note that in practice we compute the NEGATIVE likelihood as that is more convenient
        loss_mc = torch.zeros(minibatch_size, device=self.get_device())
        for _ in range(self.mc_samples):
            # Sample a mask from the variational distribution
            mask = self.variational_family.sample(minibatch_size).detach()

            # Compute model's predicted probabilities
            class_scores = model(minibatch, mask)
            assert class_scores.shape == (minibatch_size, num_values)

            # Compute negative log-likelihood
            loss_mc += loss_fn(class_scores, minibatch_assignment)

        # Average each loss across samples
        loss_mc /= self.mc_samples

        # Obtain expected loss per datapoint
        loss_mc = loss_mc.sum() / minibatch_size


        ######### REINFORCE
        loss_rf = torch.zeros(minibatch_size, device=self.get_device())
        loss_rf_real = torch.zeros(minibatch_size, device=self.get_device())
        for _ in range(self.mc_samples):
            # Sample a mask from the variational distribution
            mask = self.variational_family.sample(minibatch_size).detach()

            # Compute model's predicted probabilities
            class_scores = model(minibatch, mask)
            assert class_scores.shape == (minibatch_size, num_values)

            # Compute negative log-likelihood
            loss_rf += loss_fn(class_scores, minibatch_assignment).detach() * self.variational_family.logprob(mask)
            loss_rf_real += loss_fn(class_scores, minibatch_assignment).detach()

        # Average each loss across samples
        loss_rf /= self.mc_samples
        loss_rf_real /= self.mc_samples

        # Obtain expected loss per datapoint
        loss_rf = loss_rf.sum() / minibatch_size
        loss_rf_real = loss_rf_real.mean()

        # COMPUTE SECOND TERM OF LOSS, AKA., ENTROPY
        # We make it negative since we are minimizing, not maximizing, the objective
        loss_entropy = -self.variational_family.entropy()

        # COMPUTE REGULARIZATION
        l1_weight, l2_weight = self._l1_weight, self._l2_weight

        weights_regularization: Union[torch.Tensor, float] = 0.0
        for p in model.parameters():
            weights_regularization += l1_weight * p.abs().sum() + l2_weight * (p ** 2).sum()

        loss = loss_mc + loss_entropy + loss_rf + weights_regularization
        # The actual likelihood
        loss_real = loss_mc + self.entropy_scale * loss_entropy + weights_regularization

        # The first loss is used for backpropagation.
        # The second loss is the "correct" one, used for early stopping.
        return loss, loss_real
