import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


def get_mnist_train_data(n=-1):
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将数据转换为 [0, 1] 范围
        # transforms.Normalize((0.5,), (0.5,))  # 如果需要标准化到 [-1, 1]，可以取消注释
    ])

    # 数据集路径
    download_path = '/home/database'

    # 加载 MNIST 数据集
    train_dataset = datasets.MNIST(
        root=download_path, train=True, download=False, transform=transform)

    # 获取数据和标签
    data = train_dataset.data
    labels = train_dataset.targets

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
        sampled_indices = sampled_indices[torch.randperm(len(sampled_indices))]  # 随机打乱

    # 获取采样后的数据和标签
    sampled_labels = labels[sampled_indices]
    sampled_images = data[sampled_indices].float()  # 转换为浮点数

    # 手动应用 transform
    sampled_images = sampled_images / 255.0  # 将 [0, 255] 转换为 [0, 1]

    # 如果需要将图像展平为 [n, 28*28]
    sampled_images = sampled_images.view(-1, 28*28)

    return sampled_images, sampled_labels


def get_mnist_test_data(n=-1):
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将数据转换为 [0, 1] 范围
        # transforms.Normalize((0.5,), (0.5,))  # 如果需要标准化到 [-1, 1]，可以取消注释
    ])

    # 数据集路径
    download_path = '/home/database'

    # 加载 MNIST 数据集
    test_dataset = datasets.MNIST(
        root=download_path, train=False, download=False, transform=transform)

    # 获取数据和标签
    data = test_dataset.data
    labels = test_dataset.targets

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
        sampled_indices = sampled_indices[torch.randperm(len(sampled_indices))]  # 随机打乱

    # 获取采样后的数据和标签
    sampled_labels = labels[sampled_indices]
    sampled_images = data[sampled_indices].float()  # 转换为浮点数

    # 手动应用 transform
    sampled_images = sampled_images / 255.0  # 将 [0, 255] 转换为 [0, 1]

    # 如果需要将图像展平为 [n, 28*28]
    sampled_images = sampled_images.view(-1, 28*28)

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

class MnistFeatureDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.fixed_label = 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            # 处理切片情况
            start, stop, step = index.indices(len(self.data))
            sliced_data = self.data[start:stop:step]
            # 生成与切片数据长度匹配的标签列表
            num_samples = len(sliced_data)
            sliced_labels = torch.full((num_samples,), self.fixed_label, dtype=torch.long)
            return sliced_data, sliced_labels
        else:
            # 处理单索引情况
            sample = self.data[index]
            return sample, self.fixed_label


def get_mnist_feature_dataset(n=-1):
    data, labels = get_mnist_train_data(n)
    dataset = MnistFeatureDataset(data.T)
    return dataset, labels
