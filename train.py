import os
import random
from datetime import datetime

import torch

from data import cifar_loader, mnist_loader
from data.archive import gene_loader
from model.cnn_embedder import CnnEmbedder
from model.configs.cifar_config import CifarConfig
from model.configs.gene_config import GeneFeatureConfig, GeneSampleConfig
from model.configs.mnist_config import (MiniMnistConfig, MnistCnnConfig,
                                        MnistConfig, MnistFeatureConfig)
from model.flow_embedder import FlowEmbedder
from model.modules.advanced.vae import AeCnn
from model.mlp_embedder import MlpEmbedder
from train.vae_embedder_trainer import FeatureTrainer, SampleTrainer
from train.gene_trainer import GeneSampleTrainer, GeneFeatureTrainer
from train.mnist_trainer import MnistSampleTrainer
from train.vae_trainer import VaeTrainer
from utils.data_utils import setup_seed


def generate_log_dir(*dirs):
    current_dir = os.getcwd()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = random.randint(1000, 9999)  # 添加随机数
    unique_timestamp = f"{timestamp}_{unique_id}"

    log_dir = os.path.join(current_dir, "log", "train",
                           *dirs, unique_timestamp)

    os.makedirs(log_dir, exist_ok=True)

    return log_dir


def train_mnist_mlp_sample(seed, device, dtype):
    config = MnistConfig()

    log_dir = generate_log_dir("MNIST", "MLP", "sample")

    with open(os.path.join(log_dir, f"{seed}.txt"), "w") as f:
        pass

    # images, labels = mnist_loader.get_mnist_train_data(6000)
    data = mnist_loader.get_mnist_dataset(6000)

    torch.save(data.data, os.path.join(log_dir, "mnist.pth"))
    torch.save(data.labels, os.path.join(log_dir, "labels.pth"))

    model = MlpEmbedder(config=config)

    # trainer = GeneSampleTrainer(data=images, model=model,
    #                         log_dir=log_dir, labels=labels)
    trainer = MnistSampleTrainer(dataset=data, model=model,
                            log_dir=log_dir, device=device, dtype=dtype)

    trainer.train()


def train_mnist_mlp_feature(device, dtype):
    config = MnistFeatureConfig()

    log_dir = generate_log_dir("MNIST", "MLP", "feature")

    # images, labels = mnist_loader.get_mnist_train_data(1000)
    data, labels = mnist_loader.get_mnist_feature_dataset(1000)
    # print(data[:])
    # return
    torch.save(data.data.T, os.path.join(log_dir, "mnist.pth"))
    torch.save(labels, os.path.join(log_dir, "labels.pth"))

    model = MlpEmbedder(config=config)

    # trainer = GeneSampleTrainer(data=images, model=model,
    #                         log_dir=log_dir, labels=labels)
    trainer = FeatureTrainer(dataset=data, model=model,
                            log_dir=log_dir, device=device, dtype=dtype)

    trainer.train()


def train_gene_mlp_sample(device, dtype):
    config = GeneSampleConfig()

    log_dir = generate_log_dir("GENE", "MLP", "sample")

    gene = gene_loader.get_gene_dataset()

    model = MlpEmbedder(config=config)

    trainer = GeneSampleTrainer(dataset=gene, model=model,
                            log_dir=log_dir, device=device, dtype=dtype)

    trainer.train()


def train_gene_mlp_feature(device, dtype):
    config = GeneFeatureConfig()

    log_dir = generate_log_dir("GENE", "MLP", "feature")

    gene = gene_loader.get_gene_feature_dataset()

    model = MlpEmbedder(config=config)

    trainer = GeneFeatureTrainer(dataset=gene, model=model,
                            log_dir=log_dir, device=device, dtype=dtype)

    trainer.train()


def train_cifar_mlp_sample(seed, device, dtype):
    config = CifarConfig()

    log_dir = generate_log_dir("CIFAR", "MLP", "sample")

    images = cifar_loader.get_cifar_dataset()

    model = MlpEmbedder(config=config)

    trainer = SampleTrainer(dataset=images, model=model, log_dir=log_dir, device=device, dtype=dtype)

    trainer.train()


if __name__ == "__main__":
    seed = 43
    setup_seed(seed)
    train_cifar_mlp_sample(seed=seed, device="cuda:0", dtype=torch.float64)
    