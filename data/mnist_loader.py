import torch
from torch.utils.data import Dataset

MNIST_FEATURE_PATH = '/home/database/mnist/features.pth'
MNIST_LABEL_PATH = '/home/database/mnist/labels.pth'


def get_mnist_train_data(n=-1):
    data = torch.load(MNIST_FEATURE_PATH)
    labels = torch.load(MNIST_LABEL_PATH)

    if n > 0:
        class_indices = [torch.where(labels == i)[0] for i in range(10)]

        samples_per_class = n // 10

        sampled_indices = []
        for indices in class_indices:
            if len(indices) > samples_per_class:
                sampled_indices.append(
                    indices[torch.randperm(len(indices))[:samples_per_class]])
            else:
                sampled_indices.append(indices)

        sampled_indices = torch.cat(sampled_indices)

        sampled_indices = sampled_indices[torch.randperm(len(sampled_indices))]
    else:
        sampled_indices = torch.arange(len(data))

    sampled_labels = labels[sampled_indices]
    sampled_images = data[sampled_indices].float()

    return sampled_images, sampled_labels


class MnistDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_mnist_dataset(n=-1):
    data, labels = get_mnist_train_data(n)
    dataset = MnistDataset(data, labels)
    return dataset
