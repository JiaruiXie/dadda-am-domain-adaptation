import torch
import torch.nn.functional as F


def domain_adversarial_loss(
    source_domain_predictions,
    target_domain_predictions,
):
    """
    Domain classification loss.

    Used for:
    - DANN
    - DADDA
    """

    source_labels = torch.zeros_like(
        source_domain_predictions
    )

    target_labels = torch.ones_like(
        target_domain_predictions
    )

    source_loss = F.binary_cross_entropy(
        source_domain_predictions,
        source_labels,
    )

    target_loss = F.binary_cross_entropy(
        target_domain_predictions,
        target_labels,
    )

    total_loss = (
        source_loss +
        target_loss
    )

    return total_loss