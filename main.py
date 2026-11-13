"""
main entry point for running an algorithm on Vantage6
"""

from vantage6.client import UserClient as Client
# from pathlib import Path
import json
import credentials
# import numpy as np
import time

def main(client: Client, input: dict, collab: int):
    # Run the central method on 1 node and get the results (assuming we have one collab, while we're testing like this)
    # collab = client.collaboration.list()
    # print(f'collab: {collab}')
    # ['id']


    # collab = client.collaboration.list()['data'][0]['id']

    task = client.task.create(
        input_= input,
        organizations=[2,3], # TODO: un-hardcode orgs
        # organizations=[3],
        image = credentials.algorithm_image,
        collaboration = 1,
        name="onion_central",
        description="central part of onion (top level)"
    )

    # print(task)
    results = client.wait_for_results(task.get("id"))
    # print(results)

    return results




if __name__ == "__main__":
    # local_config_path = Path(__file__).resolve().parents[1] / "config.json"

    # # read in config file
    # with open(local_config_path, "r") as fp:
    #     config = json.load(fp)
    #TODO: make this file-based (see above)
    config = {
            "num_rounds": 3,
            "model" : "base",
            "organizations": [2,3],
            "participating_orgs": [2,3],
            "seed" : 7,
            "current_level" : -1,
            "algorithm_image": credentials.algorithm_image,
            "encryption": None
        }


    client = Client(
        credentials.server_url, credentials.server_port, credentials.server_api, log_level = 'debug'
    )
    client.authenticate(credentials.username, credentials.password)
    # client.setup_encryption(credentials.private_key)
    
    organizations = client.organization.list()['data']
    # selected_orgs = np.array(organizations)[[org['name'].startswith('ARGOS') for org in organizations]]


    # setup vantage stuff
    task_input={
        "method":"central",
        # "method" : "test_gpu",
        "kwargs": {
            "config": config,
        },
    }
    # collab = client.collaboration.list()['data'][0]['id']
    collab = client.collaboration.get(id_= 1) # 1 = top level, 2 = flat
    # print(collab)    

    # print(f'config: {config} ')

    # print(f'participating org ids: {participating_org_ids}')

    # print(f'selected org ids: {selected_org_ids}')

    results = main(client, task_input, collab)

    with open(f'onion_results_{time.time()}.json', "w") as fp:
        json.dump(results, fp)