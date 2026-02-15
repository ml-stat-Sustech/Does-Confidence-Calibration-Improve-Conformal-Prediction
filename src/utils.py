"""
Utility functions including:
- Random seed setup and Registry mechanism
- Model loading (ResNet, DenseNet, VGG, ViT)
- Data loading (ImageNet dataloader)
- Evaluation metrics registration and computation
- Score function and preprocessor builders
"""

import os
import random
import warnings
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

__all__ = [
    "Registry",
    "set_seed",
    "build_model",
    "build_dataloader_imagenet",
    "split_logits_labels",
    "get_device",
    "build_score",
    "build_preprocessor",
]


# =============================================================================
# Registry Mechanism
# =============================================================================

class Registry:
    """A registry providing name -> object mapping, to support custom modules.

    Example:
        >>> BACKBONE_REGISTRY = Registry('BACKBONE')
        >>> @BACKBONE_REGISTRY.register()
        ... class MyBackbone(nn.Module):
        ...     pass
    """

    def __init__(self, name):
        self._name = name
        self._obj_map = dict()

    def _do_register(self, name, obj, force=False):
        if name in self._obj_map and not force:
            raise KeyError(
                f'An object named "{name}" was already registered in "{self._name}" registry'
            )
        self._obj_map[name] = obj

    def register(self, obj=None, force=False):
        if obj is None:
            # Used as a decorator
            def wrapper(fn_or_class):
                name = fn_or_class.__name__
                self._do_register(name, fn_or_class, force=force)
                return fn_or_class
            return wrapper
        # Used as a function call
        name = obj.__name__
        self._do_register(name, obj, force=force)

    def get(self, name):
        if name not in self._obj_map:
            raise KeyError(
                f'Object name "{name}" does not exist in "{self._name}" registry'
            )
        return self._obj_map[name]

    def registered_names(self):
        return list(self._obj_map.keys())


METRICS_REGISTRY = Registry("METRICS")


# =============================================================================
# Random Seed
# =============================================================================

def set_seed(seed):
    """Set global random seed for reproducibility."""
    if seed != 0:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


# =============================================================================
# Model Loading
# =============================================================================

def build_model(model_name):
    """Load pretrained model."""
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    elif model_name == "densenet121":
        model = models.densenet121(weights="IMAGENET1K_V1", progress=True)
    elif model_name == "vgg16":
        model = models.vgg16(weights="IMAGENET1K_V1", progress=True)
    elif model_name == "vit":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    model.cuda()
    cudnn.benchmark = True
    model.eval()
    return model


# =============================================================================
# Data Loading
# =============================================================================

# ImageNet normalization parameters
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

transform_imagenet_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_dataloader_imagenet(data_dir, conf_num, temp_num, batch_size, num_workers):
    """
    Build ImageNet dataloaders.
    
    Splits validation set into three parts:
    - calib_calibloader: for temperature calibration
    - conf_calibloader: for conformal prediction threshold computation
    - testloader: for test evaluation
    """
    validir = os.path.join(data_dir, 'imagenet/images/val')
    testset = datasets.ImageFolder(root=validir, transform=transform_imagenet_test)

    dataset_length = len(testset)
    cal_num = conf_num + temp_num
    calibset, testset = torch.utils.data.random_split(testset, [cal_num, dataset_length - cal_num])
    conf_calibset, calib_calibset = torch.utils.data.random_split(calibset, [conf_num, cal_num - conf_num])

    calib_calibloader = torch.utils.data.DataLoader(
        dataset=calib_calibset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    conf_calibloader = torch.utils.data.DataLoader(
        dataset=conf_calibset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True
    )

    return calib_calibloader, conf_calibloader, testloader


# =============================================================================
# Utility Functions
# =============================================================================

def split_logits_labels(model, dataloader):
    """Extract logits and labels from dataloader."""
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.cuda()
            labels = labels.cuda()
            logits = model(images)
            logits_list.append(logits)
            labels_list.append(labels)

        logits_list = torch.cat(logits_list).cuda()
        labels_list = torch.cat(labels_list).cuda()
    return logits_list, labels_list


def get_device(model=None):
    """Get computing device."""
    if model is None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            device = torch.device(f"cuda:{cuda_idx}")
    else:
        device = next(model.parameters()).device
    return device


# =============================================================================
# Evaluation Metrics
# =============================================================================

@METRICS_REGISTRY.register()
def coverage_rate(prediction_sets, labels):
    """Compute coverage rate: proportion of true labels in prediction sets."""
    cvg = 0
    for ele in zip(prediction_sets, labels):
        if ele[1] in ele[0]:
            cvg += 1
    return cvg / len(prediction_sets)


@METRICS_REGISTRY.register()
def average_size(prediction_sets, labels):
    """Compute average prediction set size."""
    avg_size = 0
    for ele in prediction_sets:
        avg_size += len(ele)
    return avg_size / len(prediction_sets)


@METRICS_REGISTRY.register()
def accuracy(probs, targets, top_k=(1,)):
    """Compute Top-K accuracy."""
    k_max = max(top_k)
    batch_size = targets.size(0)

    _, order = probs.topk(k_max, dim=1, largest=True, sorted=True)
    order = order.t()
    correct = order.eq(targets.view(1, -1).expand_as(order))

    acc = []
    for k in top_k:
        correct_k = correct[:k].float().sum()
        acc.append(correct_k.mul_(100.0 / batch_size))
    return acc[0].item()


class Metrics:
    """Metrics caller, get metric function by name."""

    def __call__(self, metric) -> Any:
        if metric not in METRICS_REGISTRY.registered_names():
            raise NameError(f"The metric '{metric}' is not defined.")
        return METRICS_REGISTRY.get(metric)


class DimensionError(Exception):
    pass


# =============================================================================
# Score Functions Builder
# =============================================================================

def build_score(conformal, penalty=None):
    """Build score function by name."""
    if conformal == "aps":
        from src.conformal import APS
        return APS()
    elif conformal == "raps":
        from src.conformal import RAPS
        return RAPS(penalty=penalty)
    else:
        raise NotImplementedError(f"Score function '{conformal}' is not implemented.")


# =============================================================================
# Preprocessor Builder
# =============================================================================

def build_preprocessor(method, model=None, alpha=None):
    """
    Build preprocessor by method name.
    
    Args:
        method: Preprocessor method name
            - 'confts': Conformal Temperature Scaling
            - 'confps': Conformal Platt Scaling  
            - 'confvs': Conformal Vector Scaling
            - 'ts': Temperature Scaling
            - 'ps': Platt Scaling
            - 'vs': Vector Scaling
            - 'none' or others: Identity
        model: Model instance (required for conformal methods)
        alpha: Significance level (required for conformal methods)
    
    Returns:
        Preprocessor instance
    """
    from src.scalings import (
        Identity, TemperatureScaling, PlattScaling, VectorScaling,
        ConformalTemperatureScaling, ConformalPlattScaling, ConformalVectorScaling
    )
    
    method = method.lower()
    if method == 'confts':
        if model is None or alpha is None:
            raise ValueError("ConformalTemperatureScaling requires model and alpha")
        return ConformalTemperatureScaling(model, alpha)
    elif method == 'confps':
        if model is None or alpha is None:
            raise ValueError("ConformalPlattScaling requires model and alpha")
        return ConformalPlattScaling(model, alpha)
    elif method == 'confvs':
        if model is None or alpha is None:
            raise ValueError("ConformalVectorScaling requires model and alpha")
        return ConformalVectorScaling(model, alpha)
    elif method == 'ts':
        return TemperatureScaling()
    elif method == 'ps':
        return PlattScaling()
    elif method == 'vs':
        return VectorScaling()
    else:
        return Identity()
