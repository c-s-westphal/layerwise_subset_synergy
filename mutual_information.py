import numpy as np
from sklearn.metrics import mutual_info_score


def calculate_mutual_information(labels: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate mutual information between true labels and predictions.

    MI(Y; Y_hat) measures how much information the predictions provide about the true labels.

    Args:
        labels: True labels (ground truth)
        predictions: Model predictions

    Returns:
        Mutual information score (in nats if using natural logarithm)
    """
    return mutual_info_score(labels, predictions)


def calculate_average_mi(all_predictions: list, labels: np.ndarray) -> float:
    """
    Calculate average mutual information across multiple prediction sets.

    Args:
        all_predictions: List of prediction arrays, one per masked model
        labels: True labels (same for all predictions)

    Returns:
        Average mutual information across all prediction sets
    """
    mi_scores = []

    for predictions in all_predictions:
        mi = calculate_mutual_information(labels, predictions)
        mi_scores.append(mi)

    return np.mean(mi_scores), np.std(mi_scores)


def calculate_mi_per_layer(predictions_by_layer: dict, labels: np.ndarray) -> dict:
    """
    Calculate average MI for each layer.

    Args:
        predictions_by_layer: Dict mapping layer_idx -> list of prediction arrays
        labels: True labels

    Returns:
        Dict mapping layer_idx -> (mean_mi, std_mi)
    """
    mi_by_layer = {}

    for layer_idx, predictions_list in predictions_by_layer.items():
        mean_mi, std_mi = calculate_average_mi(predictions_list, labels)
        mi_by_layer[layer_idx] = (mean_mi, std_mi)

    return mi_by_layer
