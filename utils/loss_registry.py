# utils/loss_registry.py
import torch.nn as nn

LOSS_FUNCTIONS = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "bce": nn.BCELoss,
    "bce_with_logits": nn.BCEWithLogitsLoss,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
    "nll": nn.NLLLoss,
    "hinge_embedding": nn.HingeEmbeddingLoss,
    "cosine_embedding": nn.CosineEmbeddingLoss,
    "ctc": nn.CTCLoss,
    "huber": nn.HuberLoss,
}
