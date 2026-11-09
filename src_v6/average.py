import numpy as np


def average(weight_list, sample_sizes):
    """Sample-size-weighted (FedAvg) average of per-client model weights.

    Parameters
    ----------
    weight_list : list[dict[str, list]]
        One weight dict per client, keys are parameter names (as returned by
        ``model.get_weights()`` / ``train_locally``), values are nested lists.
    sample_sizes : list
        Number of samples each client trained on, in the same order as
        ``weight_list``.

    Returns
    -------
    dict[str, list]
        Weight dict with the same keys, values weighted-averaged and converted
        to plain lists so the result stays JSON-serializable.
    """
    if not weight_list:
        raise ValueError("No weights to average")
    if len(weight_list) != len(sample_sizes):
        raise ValueError("weight_list and sample_sizes must have the same length")

    weights = np.asarray(sample_sizes, dtype=np.float64)
    total = float(weights.sum())
    if total == 0:
        raise ValueError("sample_sizes sum to zero")

    averaged = {}
    for name in weight_list[0]:
        stacked = np.array([w[name] for w in weight_list], dtype=np.float64)
        averaged[name] = np.average(stacked, axis=0, weights=weights).tolist()
    return averaged
