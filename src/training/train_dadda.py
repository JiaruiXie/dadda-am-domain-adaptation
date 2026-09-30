import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from src.losses.discrepancy import (
    discrepancy_loss,
)

from src.losses.domain_adversarial import (
    domain_adversarial_loss,
)


# ============================================================
# DADDA Training Function
# ============================================================

def train_dadda(
    model,
    source_loader,
    target_loader,
    optimizer_encoder,
    optimizer_task_classifiers,
    optimizer_domain_classifier,
    classification_criterion,
    device,
    num_epochs=100,
    lambda_discrepancy=1.0,
    lambda_domain=1.0,
):

    model.to(device)

    history = {
        "classification_loss": [],
        "discrepancy_loss": [],
        "domain_loss": [],
        "total_loss": [],
    }

    for epoch in range(num_epochs):

        model.train()

        epoch_classification_loss = 0.0
        epoch_discrepancy_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_total_loss = 0.0

        target_iterator = iter(target_loader)

        progress_bar = tqdm(
            source_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )

        for source_images, source_labels in progress_bar:

            try:

                target_images, _ = next(
                    target_iterator
                )

            except StopIteration:

                target_iterator = iter(
                    target_loader
                )

                target_images, _ = next(
                    target_iterator
                )

            source_images = source_images.to(device)
            source_labels = source_labels.to(device)

            target_images = target_images.to(device)

            # ====================================================
            # Forward Pass
            # ====================================================

            outputs = model(
                source_images=source_images,
                target_images=target_images,
            )

            source_logits_1 = outputs[
                "source_logits_1"
            ]

            source_logits_2 = outputs[
                "source_logits_2"
            ]

            target_logits_1 = outputs[
                "target_logits_1"
            ]

            target_logits_2 = outputs[
                "target_logits_2"
            ]

            source_domain_predictions = outputs[
                "source_domain_predictions"
            ]

            target_domain_predictions = outputs[
                "target_domain_predictions"
            ]

            # ====================================================
            # Classification Loss
            # ====================================================

            classification_loss_1 = (
                classification_criterion(
                    source_logits_1,
                    source_labels,
                )
            )

            classification_loss_2 = (
                classification_criterion(
                    source_logits_2,
                    source_labels,
                )
            )

            classification_loss = (
                classification_loss_1 +
                classification_loss_2
            )

            # ====================================================
            # Discrepancy Loss
            # ====================================================

            discrepancy = discrepancy_loss(
                target_logits_1,
                target_logits_2,
            )

            # ====================================================
            # Domain Adversarial Loss
            # ====================================================

            domain_loss = (
                domain_adversarial_loss(
                    source_domain_predictions,
                    target_domain_predictions,
                )
            )

            # ====================================================
            # Total Loss
            # ====================================================

            total_loss = (
                classification_loss
                + lambda_discrepancy * discrepancy
                + lambda_domain * domain_loss
            )

            # ====================================================
            # Backpropagation
            # ====================================================

            optimizer_encoder.zero_grad()

            optimizer_task_classifiers.zero_grad()

            optimizer_domain_classifier.zero_grad()

            total_loss.backward()

            optimizer_encoder.step()

            optimizer_task_classifiers.step()

            optimizer_domain_classifier.step()

            # ====================================================
            # Record Losses
            # ====================================================

            epoch_classification_loss += (
                classification_loss.item()
            )

            epoch_discrepancy_loss += (
                discrepancy.item()
            )

            epoch_domain_loss += (
                domain_loss.item()
            )

            epoch_total_loss += (
                total_loss.item()
            )

            progress_bar.set_postfix({

                "classification":
                    classification_loss.item(),

                "discrepancy":
                    discrepancy.item(),

                "domain":
                    domain_loss.item(),

                "total":
                    total_loss.item(),
            })

        # ========================================================
        # Epoch Statistics
        # ========================================================

        num_batches = len(source_loader)

        epoch_classification_loss /= num_batches

        epoch_discrepancy_loss /= num_batches

        epoch_domain_loss /= num_batches

        epoch_total_loss /= num_batches

        history["classification_loss"].append(
            epoch_classification_loss
        )

        history["discrepancy_loss"].append(
            epoch_discrepancy_loss
        )

        history["domain_loss"].append(
            epoch_domain_loss
        )

        history["total_loss"].append(
            epoch_total_loss
        )

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Classification: {epoch_classification_loss:.4f} | "
            f"Discrepancy: {epoch_discrepancy_loss:.4f} | "
            f"Domain: {epoch_domain_loss:.4f} | "
            f"Total: {epoch_total_loss:.4f}"
        )

    return history