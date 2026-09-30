import torch


def gaussian_kernel(
    source,
    target,
    kernel_mul=2.0,
    kernel_num=5,
    fix_sigma=None,
):
    """
    Compute Gaussian kernel matrix.
    """

    n_samples = int(
        source.size(0) +
        target.size(0)
    )

    total = torch.cat(
        [source, target],
        dim=0,
    )

    total_0 = total.unsqueeze(0).expand(
        total.size(0),
        total.size(0),
        total.size(1),
    )

    total_1 = total.unsqueeze(1).expand(
        total.size(0),
        total.size(0),
        total.size(1),
    )

    L2_distance = (
        (total_0 - total_1) ** 2
    ).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma

    else:
        bandwidth = torch.sum(
            L2_distance.data
        ) / (
            n_samples ** 2 -
            n_samples
        )

    bandwidth /= kernel_mul ** (
        kernel_num // 2
    )

    bandwidth_list = [
        bandwidth * (kernel_mul ** i)
        for i in range(kernel_num)
    ]

    kernel_values = [
        torch.exp(
            -L2_distance / bandwidth_temp
        )
        for bandwidth_temp in bandwidth_list
    ]

    return sum(kernel_values)


def mmd_loss(
    source_features,
    target_features,
):
    """
    Maximum Mean Discrepancy loss.

    Used for:
    - DeepMMD
    """

    batch_size = source_features.size(0)

    kernels = gaussian_kernel(
        source_features,
        target_features,
    )

    XX = kernels[
        :batch_size,
        :batch_size,
    ]

    YY = kernels[
        batch_size:,
        batch_size:,
    ]

    XY = kernels[
        :batch_size,
        batch_size:,
    ]

    YX = kernels[
        batch_size:,
        :batch_size,
    ]

    loss = torch.mean(
        XX + YY - XY - YX
    )

    return loss