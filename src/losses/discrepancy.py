import torch
import torch.nn.functional as F


def discrepancy_loss(
    logits_1,
    logits_2,
):
    """
    Symmetric KL divergence discrepancy loss.

    Used for:
    - DDA
    - DADDA

    Encourages class-conditional alignment
    between dual task classifiers.
    """

    probabilities_1 = F.softmax(
        logits_1,
        dim=1,
    )

    probabilities_2 = F.softmax(
        logits_2,
        dim=1,
    )

    log_probabilities_1 = torch.log(
        probabilities_1 + 1e-8
    )

    log_probabilities_2 = torch.log(
        probabilities_2 + 1e-8
    )

    kl_1_to_2 = F.kl_div(
        log_probabilities_1,
        probabilities_2,
        reduction="batchmean",
    )

    kl_2_to_1 = F.kl_div(
        log_probabilities_2,
        probabilities_1,
        reduction="batchmean",
    )

    symmetric_kl = (
        kl_1_to_2 +
        kl_2_to_1
    ) / 2.0

    return symmetric_kl