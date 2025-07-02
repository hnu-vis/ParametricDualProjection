import os
import random
from datetime import datetime

import torch

from data import cifar_loader, mnist_loader
from model.configs.cifar_config import CifarConfig
from model.configs.mnist_config import MnistConfig
from model.mlp_embedder import MlpEmbedder
from train.mlp_embedder_trainer import SampleTrainer
from utils.data_utils import setup_seed


def generate_log_dir(*dirs):
    current_dir = os.getcwd()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = random.randint(1000, 9999)
    unique_timestamp = f"{timestamp}_{unique_id}"

    log_dir = os.path.join(current_dir, "log", "train",
                           *dirs, unique_timestamp)

    os.makedirs(log_dir, exist_ok=True)

    return log_dir


def train_mnist_mlp_sample(seed, device, dtype):
    config = MnistConfig()

    log_dir = generate_log_dir("MNIST", "MLP", "sample")

    open(os.path.join(log_dir, f"{seed}.txt"), "w").close()

    dataset = mnist_loader.get_mnist_dataset()

    model = MlpEmbedder(config=config)

    trainer = SampleTrainer(dataset=dataset, model=model,
                                 log_dir=log_dir, device=device, dtype=dtype)

    trainer.train()


def train_cifar_mlp_sample(seed, device, dtype):
    config = CifarConfig()

    log_dir = generate_log_dir("CIFAR", "MLP", "sample")

    open(os.path.join(log_dir, f"{seed}.txt"), "w").close()

    dataset = cifar_loader.get_cifar_dataset()

    model = MlpEmbedder(config=config)

    trainer = SampleTrainer(dataset=dataset, model=model,
                            log_dir=log_dir, device=device, dtype=dtype)

    trainer.train()


if __name__ == "__main__":
    for i in range(10):
        seeds = {}
        seed = random.randint(30, 130)

        while seed in seeds:
            seed = random.randint(30, 130)

        setup_seed(seed)
        train_cifar_mlp_sample(seed=seed, device="cuda:0", dtype=torch.float64)
