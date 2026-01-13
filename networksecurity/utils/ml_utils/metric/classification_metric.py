import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import ClassficationMetricArtifact
from sklearn.metrics import f1_score, precision_score, recall_score


def get_classification_score(y_true, y_pred) -> ClassficationMetricArtifact:
    """
    Calculate classification metrics: f1 score, precision, recall.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        ClassficationMetricArtifact: An artifact containing the calculated metrics.
    """
    try:
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        classification_metric_artifact = ClassficationMetricArtifact(
            f1_score=f1,
            precision_score=precision,
            recall_score=recall
        )

        logging.info(f"Classification metrics calculated: {classification_metric_artifact}")
        return classification_metric_artifact

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e