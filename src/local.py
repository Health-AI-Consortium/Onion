import os
from typing import Any

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from vantage6.client import UserClient as Client

from src.central import central
from src.model import get_model
from test.test_data.load_MNIST import get_MNIST_partition


def local(config: dict, model_weights: Any):

    if os.environ['END_NODE'] == "TRUE":
        return train_locally(config, model_weights)
    else:
        return sub_central(config, model_weights)



def sub_central(config, model_weights):
    # this is the bridge between networks
    client = Client(os.environ['SERVER_URL'], os.environ['SERVER_PORT'], os.environ['SERVER_API'], log_level = 'debug')
    client.authenticate(os.environ['USERNAME'], os.environ['PASSWORD'])

    if config['encryption']:
        client.setup_encryption(os.environ['PRIVATE_KEY'])

    #TODO: this doesn't seem stable with multiple collabs
    collab = client.collaboration.list()['data'][0]['id']

    input = {
        "method": "central",
        "kwargs": {
            "config": config,
            "model_weights": model_weights
        }
    }

    task = client.task.create(
        input_ = input,
        image = config['algorithm_image'],
        organizations = config['organizations'], #TODO: this needs to be specific for this layer somehow
        collaboration = collab,
        name = f"{config['algorithm_image']}_central", #TODO: define which layer
        description = f"central part of {config['algorithm_image']}"
    ) 
    results = client.wait_for_results(task.get("id"))
    return results




def train_locally(config, model_weights):
    """Train on this client's local MNIST shard (proof of concept)."""
    if config is None:
        config = {}

    seed = config.get("seed", 7)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the distributed weights, or initialize fresh
    model = get_model(config)
    if model_weights:
        model.load_weights({k: torch.tensor(v) for k, v in model_weights.items()})
    model = model.to(device)

    # load this client's local data shard
    partition_id = config.get("partition_id", 0)
    num_partitions = config.get("num_partitions", 1)
    (x_train, y_train), (x_test, y_test) = get_MNIST_partition(partition_id, num_partitions, seed=seed)

    x_train = torch.from_numpy(x_train).to(torch.float32).div(255.0).unsqueeze(1)
    y_train = torch.from_numpy(y_train).to(torch.long)
    x_test = torch.from_numpy(x_test).to(torch.float32).div(255.0).unsqueeze(1)
    y_test = torch.from_numpy(y_test).to(torch.long)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.get("lr", 1e-3))

    # training loop
    epochs = config.get("epochs", 1)
    metrics = {}
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(train_dataset)
        metrics[f"train_loss_epoch_{epoch}"] = epoch_loss
        print(f"[train_locally] partition {partition_id} epoch {epoch}/{epochs} loss: {epoch_loss:.4f}")

    # evaluate on the client's held-out test shard
    model.eval()
    with torch.no_grad():
        correct = (model(x_test.to(device)).argmax(dim=1) == y_test.to(device)).sum().item()
    metrics["test_accuracy"] = correct / len(y_test)

    # convert weights to plain floats so they are JSON-serializable in the
    # vantage6 task result (Tensors are not)
    weights = {k: v.tolist() for k, v in model.get_weights().items()}
    return {"model_weights": weights, "sample_size": len(train_dataset), "metrics": metrics}
