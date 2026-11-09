import os
from typing import Any
import pandas as pd

from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

from src_v6.local import train_locally


def mock_local(config:dict, model_weights: Any):
    print("in mock local")
    print(f"end node: {os.environ['END_NODE']}")
    if os.environ['END_NODE'] == "TRUE":
        return mock_train_locally(config, model_weights)
    else:
        return mock_sub_central(config, model_weights)



def mock_sub_central(config, model_weights):
    os.environ['END_NODE'] = "TRUE" # set this to one, so 'next' layer is lowest layer

    print("in mock_sub_central")
    print(f"end node: {os.environ['END_NODE']}")
    mock_datasets = [
    [{"database": pd.DataFrame(), "db_type": "csv"}],
    [{"database": pd.DataFrame(), "db_type": "csv"}]
    ]
    client = MockAlgorithmClient(mock_datasets, module = "src_v6.mock_local")

    input_= {
    "method": "mock_local",
    "kwargs": {
        "config": config,
        "model_weights": model_weights
    }
}   
    task = client.task.create(
        input_ = input_,
        organizations = client.organizations_with_data,
        name = "mock_run_layer2"
    )

    result = client.wait_for_results(task_id = task.get("id"))

    return result

def mock_train_locally(config, model_weights):
    return train_locally(config, model_weights)
