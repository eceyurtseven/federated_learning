import os
import hydra
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from client import generate_client_fn
from dataset import prepare_dataset
from server import get_on_fit_config, get_evaluate_fn

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)

    print(len(trainloaders), len(trainloaders[0].dataset))

    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001, min_fit_clients=cfg.num_clients_per_round_fit, fraction_evaluate=0.00001, min_evaluate_clients=cfg.num_clients_per_round_eval, min_available_clients=cfg.num_clients, on_fit_config_fn=get_on_fit_config(cfg.config_fit), evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader))  



if __name__ == "__main__":
    main()