import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


def get_cifar_train_data(n=-1):
    # 获取数据和标签
    data = torch.load("/home/database/cifar/sampled_features_1000.pt")
    labels = torch.load("/home/database/cifar/sampled_labels_1000.pt")
    # labels = train_dataset.targets
    # labels = torch.tensor(labels)

    # 如果 n 是有效的正整数，按类别平均采样
    if n > 0:
        # 获取每个类别的索引
        class_indices = [torch.where(labels == i)[0] for i in range(10)]

        # 计算每个类别需要采样的数量
        samples_per_class = n // 10

        # 从每个类别中随机采样
        sampled_indices = []
        for indices in class_indices:
            if len(indices) > samples_per_class:
                sampled_indices.append(
                    indices[torch.randperm(len(indices))[:samples_per_class]])
            else:
                sampled_indices.append(indices)

        # 合并采样后的索引
        sampled_indices = torch.cat(sampled_indices)

        # 随机打乱采样后的索引
        sampled_indices = sampled_indices[torch.randperm(len(sampled_indices))]
    else:
        # 如果 n 是 -1 或其他无效值，返回整个数据集并随机打乱
        sampled_indices = torch.arange(len(data))
        # sampled_indices = sampled_indices[torch.randperm(len(sampled_indices))]  # 随机打乱

    # 获取采样后的数据和标签
    sampled_labels = labels[sampled_indices]
    sampled_images = data[sampled_indices].float()  # 转换为浮点数

    # 如果需要将图像展平为 [n, 28*28]
    # sampled_images = sampled_images.view(-1, 28*28)

    return sampled_images, sampled_labels

class CifarDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_cifar_dataset(n=-1):
    data, labels = get_cifar_train_data(n)
    dataset = CifarDataset(data, labels)
    return dataset
