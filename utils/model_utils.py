import os

import torch

from model.configs import MnistConfig, CifarConfig, GeneSampleConfig
from model.mlp_embedder import MlpEmbedder
from utils import data_utils

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
default_dtype = torch.float64


def prepare_gene_mlp_model(result_path, subfix):
    config_json = data_utils.load_json(
        os.path.join(result_path, "config.json"))
    config = GeneSampleConfig()
    config.load_config(config_json)

    model = MlpEmbedder(config=config)
    model.load_state_dict(torch.load(
        os.path.join(result_path, f"model_{subfix}.pth")))

    return model


def prepare_mnist_mlp_model(result_path, subfix):
    config_json = data_utils.load_json(
        os.path.join(result_path, "config.json"))
    config = MnistConfig()
    config.load_config(config_json)

    model = MlpEmbedder(config=config)
    model.load_state_dict(torch.load(
        os.path.join(result_path, f"model_{subfix}.pth")))

    return model


def prepare_cifar_mlp_model(result_path, subfix):
    config_json = data_utils.load_json(
        os.path.join(result_path, "config.json"))
    config = CifarConfig()
    config.load_config(config_json)

    model = MlpEmbedder(config=config)
    model.load_state_dict(torch.load(
        os.path.join(result_path, f"model_{subfix}.pth")))

    return model
