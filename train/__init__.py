import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import loss
from loss.nce_loss import ntxent_loss, refined_ntxent_loss
from model.base_embedder import BaseEmbedder
from utils import data_utils, vis_utils


class BaseTrainer:
    def __init__(
        self,
        dataset: Dataset,
        model: BaseEmbedder,
        log_dir: str,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"),
        dtype: torch.dtype = torch.float32
    ):
        self.dataset = dataset
        self.data, self.labels = dataset[:]
        self.model = model.to(device=device, dtype=dtype)
        self.log_dir = log_dir
        self.device = device
        self.dtype = dtype

        self.setup_config()
        self.setup_loss_weights()
        self.setup_training_params()

        self.setup_dataloader()
        self.setup_optimizer()
        self.setup_scheduler()

        self.curr_epoch = 0
        self.curr_batch_id = 0
        self.embedding_feature = None
        self.loss_history = {
            "total": [],
            "embedding": [],
            "auxiliary": [],
            "latent": [],
            "reconstruction": []
        }

    def setup_config(self):
        self.config = self.model.config
        self.input_dim = self.config.get_input_dim()
        self.output_dim = self.config.get_output_dim()
        self.latent_dim = self.config.get_latent_dim()

    def setup_loss_weights(self):
        self.embedding_weight = self.config.embedding_weight
        self.auxiliary_weight = self.config.auxiliary_weight
        self.latent_weight = self.config.latent_weight
        self.reconstruction_weight = self.config.reconstruction_weight

    def setup_training_params(self):
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr
        self.save_interval = self.epochs // 10

        self.warmup_epochs = self.config.warmup_epochs
        self.main_epochs = self.config.main_epochs
        self.stable_epochs = self.config.stable_epochs

    def setup_dataloader(self):
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def setup_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

    def setup_scheduler(self):
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs
        )

    def train(self):
        self.before_training()

        progress_bar = tqdm(range(self.epochs), desc="Training")
        for self.curr_epoch in progress_bar:
            self.before_epoch()
            epoch_metrics = self.run_epoch()
            self.after_epoch(epoch_metrics)

            progress_bar.set_postfix({
                "Tot.": f"{epoch_metrics['total']:.4f}",
                "Emb.": f"{epoch_metrics['embedding']:.4f}",
                "Aux.": f"{epoch_metrics['auxiliary']:.4f}",
                "Lat.": f"{epoch_metrics['latent']:.4f}",
                "Rec.": f"{epoch_metrics['reconstruction']:.4f}"
            })

        self.after_training()

    def run_epoch(self):
        epoch_metrics = {
            "total": 0.0,
            "embedding": 0.0,
            "auxiliary": 0.0,
            "latent": 0.0,
            "reconstruction": 0.0
        }

        for batch_id, (data, labels) in enumerate(self.dataloader):
            self.curr_batch_id = batch_id
            data = data.to(self.device, dtype=self.dtype)

            batch_metrics = self.training_step(data)

            for key in epoch_metrics:
                epoch_metrics[key] += batch_metrics[key]

        for key in epoch_metrics:
            epoch_metrics[key] /= len(self.dataloader)

        return epoch_metrics

    def training_step(self, data: torch.Tensor):
        self.optimizer.zero_grad()

        embedding_features, auxiliary_features, latent_features, reconstructed_data = self.model.forward(
            data)

        losses = self.compute_losses(
            data, embedding_features, auxiliary_features, latent_features, reconstructed_data
        )

        losses["total"].backward()
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def compute_losses(self, data, embedding_features, auxiliary_features, latent_features, reconstructed_data):
        return {
            "total": self.total_loss(data, embedding_features, auxiliary_features, latent_features, reconstructed_data),
            "embedding": self.embedding_loss(data, embedding_features),
            "auxiliary": self.auxiliary_loss(data, auxiliary_features),
            "latent": self.latent_loss(latent_features),
            "reconstruction": self.reconstruction_loss(data, reconstructed_data)
        }

    def total_loss(self, data, embedding_features, auxiliary_features, latent_features, reconstructed_data):
        return (
            self.embedding_weight * self.embedding_loss(data, embedding_features) +
            self.auxiliary_weight * self.auxiliary_loss(data, auxiliary_features) +
            self.latent_weight * self.latent_loss(latent_features) +
            self.reconstruction_weight *
            self.reconstruction_loss(data, reconstructed_data)
        )

    def embedding_loss(self, data, embedding_features):
        return loss.mds_loss(data, embedding_features)

    def auxiliary_loss(self, data, auxiliary_features):
        return loss.mds_loss(data, auxiliary_features)

    def latent_loss(self, latent_features):
        return loss.gaussian_loss(latent_features)

    def reconstruction_loss(self, data, reconstructed_data):
        return nn.MSELoss()(data, reconstructed_data)

    def before_training(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.model.train()

    def after_training(self):
        self.save_final()
        self.model.eval()

    def before_epoch(self):
        self.update_loss_weights()
        self.model.train()

    def after_epoch(self, metrics):
        for key in self.loss_history:
            self.loss_history[key].append(metrics[key])

        self.scheduler.step()

        if (self.curr_epoch + 1) % self.save_interval == 0:
            self.save_checkpoint()

    def update_loss_weights(self):
        pass

    def save_checkpoint(self):
        data = self.data.to(self.device, dtype=self.dtype)
        self.embedding_feature = self.model.encode(data)
        self.plot_current()
        self.save_model(f"{self.curr_epoch+1}")

    def save_final(self):
        data = self.data.to(self.device, dtype=self.dtype)
        self.embedding_feature = self.model.encode(data)
        self.plot_final()
        self.save_model("final")
        self.save_loss_history()
        self.save_config()

    def save_model(self, suffix):
        path = os.path.join(self.log_dir, f"model_{suffix}.pth")
        torch.save(self.model.state_dict(), path)

    def save_loss_history(self):
        df = pd.DataFrame(self.loss_history)
        df.to_csv(os.path.join(self.log_dir, "loss_history.csv"), index=False)

    def save_config(self):
        config = self.config.to_dict()

        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    def plot_current(self):
        self.plot_embedding(self.curr_epoch + 1)
        self.plot_loss_curve()

    def plot_final(self):
        self.plot_embedding("final")
        self.plot_loss_curve()

    def plot_embedding(self, suffix):
        embeddings = data_utils.to_numpy(self.embedding_feature)
        labels = data_utils.to_numpy(self.labels)
        vis_utils.plot_embeddings(
            embeddings,
            labels,
            discrete=True,
            save_path=os.path.join(self.log_dir, f"embedding_{suffix}.png")
        )

    def plot_loss_curve(self):
        vis_utils.plot_loss_curve(
            self.loss_history,
            save_path=os.path.join(self.log_dir, f"loss_curve.png")
        )


class ConstractiveTrainer(BaseTrainer):
    def setup_config(self):
        super().setup_config()
        self.k = self.config.k
        self.temparature = self.config.temparature

    def compute_dist(self, X, Y):
        return torch.cdist(X, Y)

    def get_positive_indices_batch(self, batch, labels=None, k=15):
        device = batch.device
        dtype = batch.dtype
        batch_size = batch.shape[0]

        if labels is not None:

            pos_indices = torch.zeros(
                batch_size, 1, device=device, dtype=torch.long)

            label_matches = (labels.unsqueeze(1) == labels.unsqueeze(0))

            label_matches.diagonal().fill_(False)

            same_label_indices = torch.where(label_matches)
            row_indices, col_indices = same_label_indices

            counts = torch.bincount(
                row_indices, minlength=batch_size)
            valid_rows = counts > 0

            if valid_rows.any():

                cum_counts = torch.cat(
                    [torch.tensor([0], device=device), counts.cumsum(0)[:-1]])

                offsets = torch.zeros_like(
                    row_indices, dtype=torch.long, device=device)
                valid_row_counts = counts[valid_rows]
                valid_row_indices = torch.arange(len(valid_rows), device=device)[
                    valid_rows]

                group_offsets = torch.randint(
                    0, valid_row_counts.max(), (valid_rows.sum(),), device=device)
                group_offsets = group_offsets % valid_row_counts
                offsets[cum_counts[valid_rows]] = group_offsets
                sample_indices = cum_counts[valid_rows] + \
                    offsets[cum_counts[valid_rows]]
                pos_indices[valid_rows] = col_indices[sample_indices].unsqueeze(
                    1)

            pos_indices[~valid_rows] = torch.arange(
                batch_size, device=device, dtype=torch.long)[~valid_rows].unsqueeze(1)

            return pos_indices
        else:

            distances = torch.cdist(batch, batch)
            _, indices = torch.topk(
                distances, k + 1, largest=False)

            neighbors = indices[:, 1:]

            rho_i = distances[:, 1].unsqueeze(1)

            sigma_i = torch.sum(
                torch.exp(-(distances[:, 1:] - rho_i)), dim=1, keepdim=True)

            p_j_given_i = torch.exp(-(distances[:, 1:] - rho_i) / sigma_i)

            p_j_given_i_full = torch.zeros(
                batch_size, batch_size, device=device, dtype=dtype)
            for i in range(batch_size):
                p_j_given_i_full[i, neighbors[i]
                                 ] = p_j_given_i[i, :k]

            p_ij = p_j_given_i_full + \
                p_j_given_i_full.transpose(
                    0, 1) - p_j_given_i_full * p_j_given_i_full.transpose(0, 1)

            p_ij = p_ij / p_ij.sum(dim=1, keepdim=True)

            pos_indices = torch.multinomial(p_ij, 1)
            return pos_indices

    def get_embeddings_for_loss(self, embeddings, pos_indices):

        x_sim_embedding = embeddings[pos_indices.squeeze()]
        return x_sim_embedding

    def batch_logits(self, x_embeddings, x_sim_embeddings, a=1.0, b=1.0):
        batch_size = x_embeddings.shape[0]

        all_embeddings = torch.cat([x_embeddings, x_sim_embeddings], dim=0)

        pairwise_distances = torch.cdist(all_embeddings, all_embeddings, p=2)
        pairwise_distances_sq = pairwise_distances ** 2
        q_ij = 1 / (1 + a * (pairwise_distances_sq ** b))

        l_pos = torch.diag(q_ij, batch_size)
        r_pos = torch.diag(q_ij, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(
            2 * batch_size, 1)

        mask = torch.ones_like(q_ij, dtype=torch.bool)
        mask.fill_diagonal_(False)
        mask[torch.arange(batch_size), torch.arange(
            batch_size, 2 * batch_size)] = False
        mask[torch.arange(batch_size, 2 * batch_size),
             torch.arange(batch_size)] = False

        negatives = q_ij[mask].view(2 * batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        return logits

    def embedding_loss(self, data, embedding_features):
        data = data.view(data.size(0), -1)
        pos_indices = self.get_positive_indices_batch(
            data, None, k=self.k)
        embedding_sim_features = self.get_embeddings_for_loss(
            embedding_features, pos_indices)
        logits = self.batch_logits(
            embedding_features, embedding_sim_features, a=1.8956058664239412, b=0.8006378441176886)

        if self.curr_epoch < self.warmup_epochs:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        elif self.curr_epoch < self.warmup_epochs + self.main_epochs:
            loss = refined_ntxent_loss(
                logits, tau=self.temparature, item_weights=self.embedding_weight)
        else:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        return loss

    def auxiliary_loss(self, data, auxiliary_features):
        data = data.view(data.size(0), -1)
        pos_indices = self.get_positive_indices_batch(
            data, None, k=self.k)
        embedding_sim_features = self.get_embeddings_for_loss(
            auxiliary_features, pos_indices)
        logits = self.batch_logits(
            auxiliary_features, embedding_sim_features, a=1.8956058664239412, b=0.8006378441176886)

        if self.curr_epoch < self.warmup_epochs:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        elif self.curr_epoch < self.warmup_epochs + self.main_epochs:
            loss = refined_ntxent_loss(
                logits, tau=self.temparature, item_weights=self.embedding_weight)
        else:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        return loss
