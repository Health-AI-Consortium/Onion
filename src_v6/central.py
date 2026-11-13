from typing import Any

import numpy as np

from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient

from src_v6.model import get_model
from src_v6.average import average



@algorithm_client
def central(client: AlgorithmClient, config: dict, model_weights = None) -> Any:



    # define organizations that participate
    participating_orgs = config['participating_orgs']


    # Initialize model
    if model_weights is None:
        model = get_model(config)
        model_weights = model.get_weights()

    metrics = {}
    # Iterative training loop
    for round in range(config['num_rounds']):

        # define local task input
        input_= {
            "method": "local",
            "kwargs": {
                "config": config,
                "model_weights": model_weights
            }
        }

        # send task to clients and wait for result
        task = client.task.create(
            input_ = input_,
            organizations=participating_orgs
        )
        task_results = client.wait_for_results(task.get("id"))

        # parse local results and average
        metrics[f'round {round}'] = {}
        local_weights = []
        local_sample_sizes = []

        #TODO: client id -> org name
        for id, result in enumerate(task_results):
            local_weights.append(result['model_weights'])
            local_sample_sizes.append(result['sample_size'])
            metrics[f'round {round}'][f'client {id}'] = result['metrics']


        model_weights = average(local_weights, local_sample_sizes)

    return {"model weights": model_weights, "sample size": np.sum(local_sample_sizes), "metrics": metrics}
        # save for next layer
        


