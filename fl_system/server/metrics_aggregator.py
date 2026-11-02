"""
Metrics Aggregation for Flower Strategy.
Computes sample-weighted averages of classification and fraud detection metrics
across distributed smart grid substation clients.
"""

from typing import Dict, List, Tuple
from flwr.common import Scalar


def aggregate_metrics_weighted(results: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """
    Compute sample-weighted average of client-reported metrics.

    Parameters
    ----------
    results : List[Tuple[int, Dict[str, Scalar]]]
        List of tuples (num_samples, metrics_dict) from each reporting client.

    Returns
    -------
    Dict[str, Scalar]
        Sample-weighted global metric dictionary.
    """
    if not results:
        return {}

    total_samples = sum(num_samples for num_samples, _ in results)
    if total_samples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "pr_auc",
        "roc_auc",
        "train_loss",
        "grad_norm",
    ]

    for num_samples, metrics in results:
        weight = num_samples / total_samples
        for k in metric_keys:
            if k in metrics:
                val = float(metrics[k])
                aggregated[k] = aggregated.get(k, 0.0) + (val * weight)

    # Convert back to Dict[str, Scalar]
    res: Dict[str, Scalar] = {k: round(v, 4) for k, v in aggregated.items()}
    res["total_samples"] = total_samples
    res["reporting_clients_count"] = len(results)
    return res
