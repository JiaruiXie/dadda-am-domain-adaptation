import torch
import torch.nn as nn

from src.models.encoder import (
    FeatureExtractor
)

from src.models.classifiers import (
    TaskClassifier,
    TaskClassifier1,
)

from src.models.domain_classifier import (
    DomainClassifier
)


class DADDA(nn.Module):

    def __init__(self):

        super(DADDA, self).__init__()

        self.feature_extractor = (
            FeatureExtractor()
        )

        self.task_classifier_1 = (
            TaskClassifier()
        )

        self.task_classifier_2 = (
            TaskClassifier1()
        )

        self.domain_classifier = (
            DomainClassifier()
        )

    def forward(
        self,
        source_images,
        target_images,
    ):

        # ====================================================
        # Feature Extraction
        # ====================================================

        source_features = (
            self.feature_extractor(
                source_images
            )
        )

        target_features = (
            self.feature_extractor(
                target_images
            )
        )

        # ====================================================
        # Task Classifiers
        # ====================================================

        source_predictions_1, source_hidden_1 = (
            self.task_classifier_1(
                source_features
            )
        )

        source_predictions_2, source_hidden_2 = (
            self.task_classifier_2(
                source_features
            )
        )

        target_predictions_1, target_hidden_1 = (
            self.task_classifier_1(
                target_features
            )
        )

        target_predictions_2, target_hidden_2 = (
            self.task_classifier_2(
                target_features
            )
        )

        # ====================================================
        # Domain Classifier
        # ====================================================

        source_domain_predictions = (
            self.domain_classifier(
                source_features
            )
        )

        target_domain_predictions = (
            self.domain_classifier(
                target_features
            )
        )

        return {

            "source_features":
                source_features,

            "target_features":
                target_features,

            "source_predictions_1":
                source_predictions_1,

            "source_predictions_2":
                source_predictions_2,

            "target_predictions_1":
                target_predictions_1,

            "target_predictions_2":
                target_predictions_2,

            "source_hidden_1":
                source_hidden_1,

            "source_hidden_2":
                source_hidden_2,

            "target_hidden_1":
                target_hidden_1,

            "target_hidden_2":
                target_hidden_2,

            "source_domain_predictions":
                source_domain_predictions,

            "target_domain_predictions":
                target_domain_predictions,
        }