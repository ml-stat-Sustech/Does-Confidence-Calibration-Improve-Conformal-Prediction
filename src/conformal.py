"""
Conformal prediction core implementation including:
- Score functions: APS, RAPS
- Predictor: threshold computation, prediction set generation, evaluation
"""

import math
import warnings

import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils import split_logits_labels, get_device, Metrics, build_score


class APS:
    """Adaptive Prediction Sets (APS) score function."""

    def __call__(self, logits, label=None, random=True):
        assert len(logits.shape) <= 2, "The dimension of logits must be less than 2."
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        if label is None:
            return self._calculate_all_label(probs, random=random)
        else:
            return self._calculate_single_label(probs, label, random=random)

    def _calculate_all_label(self, probs, random=True):
        indices, ordered, cumsum = self._sort_sum(probs)
        if random:
            U = torch.rand(probs.shape, device=probs.device)
            ordered_scores = cumsum - ordered * U
        else:
            ordered_scores = cumsum
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _sort_sum(self, probs):
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def _calculate_single_label(self, probs, label, random=True):
        indices, ordered, cumsum = self._sort_sum(probs)
        idx = torch.where(indices == label.view(-1, 1))
        idx_minus_one = (idx[0], idx[1] - 1)
        if random:
            U = torch.rand(indices.shape[0], device=probs.device)
            scores_first_rank = U * cumsum[idx]
            scores_usual = U * ordered[idx] + cumsum[idx_minus_one]
            scores = torch.where(idx[1] == 0, scores_first_rank, scores_usual)
        else:
            scores = cumsum[range(cumsum.shape[0]), label]
        return scores


class RAPS(APS):
    """
    Regularized Adaptive Prediction Sets (Angelopoulos et al., 2020)
    Paper: https://arxiv.org/abs/2009.14193

    Args:
        penalty: weight of regularization. When penalty = 0, RAPS = APS.
        kreg: rank of regularization, an integer in [0, num_labels].
    """

    def __init__(self, penalty=0.001, kreg=0, random=True):
        if penalty <= 0:
            raise ValueError("The parameter 'penalty' must be a positive value.")
        if kreg < 0:
            raise ValueError("The parameter 'kreg' must be a nonnegative value.")
        if type(kreg) != int:
            raise TypeError("The parameter 'kreg' must be an integer.")
        super().__init__()
        if penalty is None:
            penalty = 0.001
        self._penalty = penalty
        self._kreg = kreg

    def _calculate_all_label(self, probs, random=True):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape, device=probs.device)
        reg = torch.maximum(
            self._penalty * (torch.arange(1, probs.shape[-1] + 1, device=probs.device) - self._kreg),
            torch.tensor(0, device=probs.device)
        )
        ordered_scores = cumsum - ordered * U + reg
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _calculate_single_label(self, probs, label, random=True):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        reg = torch.maximum(
            self._penalty * (idx[1] + 1 - self._kreg),
            torch.tensor(0).to(probs.device)
        )
        scores_first_rank = U * ordered[idx] + reg
        idx_minus_one = (idx[0], idx[1] - 1)
        scores_usual = U * ordered[idx] + cumsum[idx_minus_one] + reg
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)


# =============================================================================
# Predictor
# =============================================================================

class Predictor(nn.Module):
    """Conformal prediction predictor with calibration and evaluation."""

    def __init__(self, model, preprocessor, conformal, alpha, random=True, penalty=0.001):
        super().__init__()
        self._model = model
        self.score_function = build_score(conformal, penalty=penalty)
        self._preprocessor = preprocessor
        self.alpha = alpha
        self.num_classes = 1000
        self._metric = Metrics()
        self._device = get_device()
        self.random = random

    def calibrate(self, calib_calibloader, conf_calibloader):
        """Calibrate the model using calibration data."""
        calib_logits, calib_labels = split_logits_labels(self._model, calib_calibloader)
        ece_before, ece_after = self._preprocessor.train(calib_logits, calib_labels)
        conf_logits, conf_labels = split_logits_labels(self._model, conf_calibloader)
        conf_logits = self._preprocessor(conf_logits, softmax=False)
        self.calculate_threshold(conf_logits, conf_labels)
        return ece_before, ece_after

    def calibrate_with_logits_labels(self, logits, labels):
        """Calibrate with precomputed logits and labels."""
        ece_before, ece_after = self._preprocessor.train(logits, labels)
        self.ece_before = ece_before
        self.ece_after = ece_after
        logits = self._preprocessor(logits, softmax=False)
        self.calculate_threshold(logits, labels)

    def calculate_threshold(self, logits, labels, random=True):
        """Calculate conformal quantile threshold."""
        alpha = self.alpha
        if alpha >= 1 or alpha <= 0:
            raise ValueError("Significance level 'alpha' must be in (0,1).")
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        scores = self.score_function(logits, labels, random=random)
        self.q_hat = self._calculate_conformal_value(scores, alpha)

    def _calculate_conformal_value(self, scores, alpha):
        """Compute conformal quantile value."""
        if len(scores) == 0:
            warnings.warn(
                "The number of scores is 0, which is invalid. "
                "To avoid program crash, the threshold is set as torch.inf."
            )
            return torch.inf
        quantile_value = math.ceil(scores.shape[0] + 1) * (1 - alpha) / scores.shape[0]

        if quantile_value > 1:
            warnings.warn(
                "The value of quantile exceeds 1. It should be a value in (0,1). "
                "To avoid program crash, the threshold is set as torch.inf."
            )
            return torch.inf

        return torch.quantile(scores, quantile_value, interpolation="higher").to(self._device)

    def predict(self, x_batch):
        """Generate prediction sets for a batch of inputs."""
        self._model.eval()
        tmp_logits = self._model(x_batch.to(self._device)).float()
        tmp_logits = self._preprocessor(tmp_logits, softmax=False).detach()
        sets, scores = self.predict_with_logits(tmp_logits)
        return sets, scores

    def predict_with_logits(self, logits, q_hat=None):
        """Generate prediction sets from logits."""
        scores = self.score_function(logits, random=self.random).to(self._device)
        if q_hat is None:
            S = self._generate_prediction_set(scores, self.q_hat)
        else:
            S = self._generate_prediction_set(scores, q_hat)
        return S, scores

    def evaluate(self, val_dataloader):
        """Evaluate the predictor on validation data."""
        prediction_sets = []
        probs_list = []
        labels_list = []
        scores_list = []
        with torch.no_grad():
            for examples in tqdm(val_dataloader):
                tmp_x, tmp_label = examples[0].to(self._device), examples[1].to(self._device)
                prediction_sets_batch, scores_batch = self.predict(tmp_x)
                target_scores_batch = scores_batch[range(tmp_label.shape[0]), tmp_label]
                prediction_sets.extend(prediction_sets_batch)
                tmp_probs = self._preprocessor(self._model(tmp_x)).detach()
                probs_list.append(tmp_probs)
                labels_list.append(tmp_label)
                scores_list.append(target_scores_batch)
        val_probs = torch.cat(probs_list)
        val_labels = torch.cat(labels_list)
        res_dict = {
            "top1": self._metric('accuracy')(val_probs, val_labels, [1]),
            "top5": self._metric('accuracy')(val_probs, val_labels, [5]),
            "Coverage_rate": self._metric('coverage_rate')(prediction_sets, val_labels),
            "Average_size": self._metric('average_size')(prediction_sets, val_labels),
        }
        return res_dict

    def _generate_prediction_set(self, scores, q_hat):
        """Generate prediction sets from scores and threshold."""
        if len(scores.shape) == 1:
            return torch.argwhere(scores <= q_hat).reshape(-1).tolist()
        else:
            return [torch.argwhere(scores[i] <= q_hat).reshape(-1).tolist() for i in range(scores.shape[0])]
