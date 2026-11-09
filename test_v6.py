from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
import pandas as pd

import os

# a bit whacky, we set this at 1 later when we're deeper into the hierarchy to switch.
# should probably do this in logic based on an architecture definition
os.environ['END_NODE'] = "FALSE"



mock_datasets = [
    [{"database": pd.DataFrame(), "db_type": "csv"}],
    [{"database": pd.DataFrame(), "db_type": "csv"}]
    ]

module = "src_v6.mock_local"

client = MockAlgorithmClient(mock_datasets, module)
config = {}
model_weights = []

input_= {
    "method": "mock_local",
    "kwargs": {
        "config": config,
        "model_weights": model_weights
    }
}

task = client.task.create(
    input_=input_,
    organizations = client.organizations_with_data,
    name = "mock_run"
)

result = client.wait_for_results(task_id = task.get("id"))

print(result)