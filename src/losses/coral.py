import torch


def compute_covariance(features):
    """
    Compute covariance matrix.
    """

    n = features.size(0)

    mean = torch.mean(
        features,
        dim=0,
        keepdim=True,
    )

    centered = features - mean

    covariance = (
        centered.t() @ centered
    ) / (n - 1)

    return covariance


def coral_loss(
    source_features,
    target_features,
):
    """
    CORAL loss.

    Used for:
    - DeepCORAL
    - CORAL-DDA
    """

    source_covariance = compute_covariance(
        source_features
    )

    target_covariance = compute_covariance(
        target_features
    )

    loss = torch.mean(
        (
            source_covariance -
            target_covariance
        ) ** 2
    )

    return loss