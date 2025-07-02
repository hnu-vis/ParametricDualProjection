import torch
import torch.nn as nn


def padding_loss(x, padding_length):
    if padding_length <= 0:
        return 0

    if padding_length >= x.size(-1):
        padding_length = x.size(-1)
        padding_part = x
    else:
        padding_part = x[padding_length:] if x.dim(
        ) == 1 else x[:, padding_length:]

    norm_factor = torch.sqrt(torch.tensor(
        float(padding_length), device=x.device))

    loss = torch.norm(padding_part) / norm_factor
    return loss


def zero_loss(X):
    padding_length = X.size(-1)
    norm_factor = torch.sqrt(torch.tensor(
        padding_length, device=X.device, dtype=X.dtype))
    loss = torch.norm(X, norm_factor)

    return loss


def orthogonal_loss(W):
    W_t_W = torch.matmul(W.T, W)
    I = torch.eye(W.shape[1], device=W.device)
    difference = W_t_W - I

    norm_factor = torch.sqrt(torch.tensor(float(W.shape[1]), device=W.device))

    frobenius_norm = torch.norm(difference, p='fro') / norm_factor

    return frobenius_norm


def lis_loss(X, Y, k=10):
    X_distances = torch.cdist(X, X, p=2)
    Y_distances = torch.cdist(Y, Y, p=2)

    norm_x = torch.sqrt(torch.tensor(
        float(X.shape[1]), device=X.device, dtype=X.dtype))
    norm_y = torch.sqrt(torch.tensor(
        float(Y.shape[1]), device=Y.device, dtype=Y.dtype))

    topk_distances, topk_indices = torch.topk(
        X_distances, k + 1, dim=1, largest=False)

    neighbors_distances_x = topk_distances[:, 1:] / norm_x
    neighbors_indices = topk_indices[:, 1:]

    neighbors_distances_y = Y_distances.gather(1, neighbors_indices) / norm_y

    loss = torch.sqrt(
        (neighbors_distances_x - neighbors_distances_y) ** 2).sum()

    return loss


def lis_loss_with_distances(X_distances, Y_distances, X_shape, Y_shape, k=10):
    norm_x = torch.sqrt(torch.tensor(
        float(X_shape[1]), device=X_distances.device))
    norm_y = torch.sqrt(torch.tensor(
        float(Y_shape[1]), device=Y_distances.device))

    topk_distances, topk_indices = torch.topk(
        X_distances, k + 1, dim=1, largest=False)

    neighbors_distances_x = topk_distances[:, 1:] / norm_x
    neighbors_indices = topk_indices[:, 1:]

    neighbors_distances_y = Y_distances.gather(1, neighbors_indices) / norm_y

    loss = torch.sqrt(
        (neighbors_distances_x - neighbors_distances_y) ** 2).sum()

    return loss


def pushaway_loss(X, k=10, B=1):
    norm_x = torch.sqrt(torch.tensor(float(X.shape[1]), device=X.device))
    distances = torch.cdist(X, X, p=2)

    topk_distances, _ = torch.topk(distances, k + 1, dim=1, largest=False)

    neighbors_distances = topk_distances[:, 1:] / norm_x

    mask = neighbors_distances < B
    masked_distances = neighbors_distances * mask

    loss = -torch.log1p(masked_distances[mask]).sum()

    return loss


def pushaway_loss_with_distances(distances, shape, k=10, B=1):
    norm_x = torch.sqrt(torch.tensor(float(shape[1]), device=distances.device))
    topk_distances, _ = torch.topk(distances, k + 1, dim=1, largest=False)

    neighbors_distances = topk_distances[:, 1:] / norm_x

    mask = neighbors_distances < B
    masked_distances = neighbors_distances * mask

    loss = -torch.log1p(masked_distances[mask]).sum()

    return loss


def compute_p_matrix(distances, beta):
    p_matrix = torch.exp(-distances * beta)
    p_matrix.fill_diagonal_(0)
    p_matrix = p_matrix / torch.sum(p_matrix, dim=1, keepdim=True)
    return p_matrix


def compute_q_matrix(distances):
    q_matrix = (1 + distances ** 2).pow(-1)
    q_matrix.fill_diagonal_(0)
    q_matrix = q_matrix / torch.sum(q_matrix)
    return q_matrix


def precompute_betas(X, perplexity=30, eps=1e-5):
    distances = torch.cdist(X, X, p=2)
    n = distances.shape[0]
    betas = torch.ones(n, device=X.device) * 1.0
    log_target = torch.log(torch.tensor(perplexity, device=X.device))

    for i in range(n):
        beta_min, beta_max = None, None
        for _ in range(50):
            p_i = compute_p_matrix(distances[i:i+1], beta=betas[i])
            entropy = -torch.sum(p_i * torch.log(p_i + eps))
            perplexity_i = torch.exp(entropy)

            # 如果当前 perplexity 接近目标值则停止搜索
            if torch.abs(perplexity_i - perplexity) < eps:
                break

            # 调整 beta，根据当前 perplexity 相对目标 perplexity 的大小
            if perplexity_i > perplexity:
                # 当前 beta 太小，需要增大
                beta_min = betas[i].clone()
                if beta_max is not None:
                    betas[i] = (betas[i] + beta_max) / 2
                else:
                    betas[i] *= 2
            else:
                # 当前 beta 太大，需要减小
                beta_max = betas[i].clone()
                if beta_min is not None:
                    betas[i] = (betas[i] + beta_min) / 2
                else:
                    betas[i] /= 2

    return betas


def tsne_loss(X, Y, betas, eps=1e-9):
    distances_x = torch.cdist(X, X, p=2)
    p_matrix = compute_p_matrix(distances_x, betas.view(-1, 1))

    p_matrix = (p_matrix + p_matrix.t()) / (2 * p_matrix.shape[0])

    distances_y = torch.cdist(Y, Y, p=2)
    q_matrix = compute_q_matrix(distances_y)

    kl_loss = torch.sum(
        p_matrix * torch.log((p_matrix + eps) / (q_matrix + eps)))

    return kl_loss


def rbf_kernel(X, Y, sigma=1.0):
    sqdist = torch.cdist(X, Y, p=2) ** 2
    return torch.exp(-sqdist / (2 * sigma ** 2))


def t_kernel(X, Y, sigma=1.0):
    D_XY = torch.cdist(X, Y, p=2) ** 2

    XX = torch.zeros(D_XY.shape, device=X.device, dtype=X.dtype)

    for a in [0.05, 0.2, 0.9]:
        XX += a**2 * (a**2 + D_XY)**-1
    return XX


def mmd_loss(X, Y, sigma=1.0):
    K_XX = t_kernel(X, X, sigma)
    K_YY = t_kernel(Y, Y, sigma)
    K_XY = t_kernel(X, Y, sigma)

    mmd = torch.mean(K_XX) + torch.mean(K_YY) - 2 * torch.mean(K_XY)

    return mmd


def compute_pair_probabilities(X, k):
    n_samples = X.shape[0]

    dist_matrix = torch.cdist(X, X)

    rho_i = torch.topk(dist_matrix, k=2, largest=False).values[:, 1]

    numerator = dist_matrix - rho_i.unsqueeze(1)

    _, knn_indices = torch.topk(dist_matrix, k=k+1, largest=False)
    knn_indices = knn_indices[:, 1:]

    knn_distances = torch.gather(dist_matrix, 1, knn_indices)
    knn_numerator = knn_distances - rho_i.unsqueeze(1)
    knn_p_j_given_i = torch.exp(-knn_numerator)

    sigma_i = knn_p_j_given_i.sum(dim=1)

    p_j_given_i = torch.exp(-numerator / sigma_i.unsqueeze(1))

    p_i_given_j = p_j_given_i.t()
    pi_j = (p_j_given_i + p_i_given_j) - p_j_given_i * p_i_given_j

    return pi_j


def estimate_gaussian_parameters(Z):
    mean = torch.mean(Z, dim=0)  # 沿批次维度计算均值
    var = torch.var(Z, dim=0, unbiased=False)  # 沿批次维度计算方差
    return mean, var


def kl_loss(mean, logvar):
    # kl = torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return kl


def gaussian_loss(Z):
    mean, var = estimate_gaussian_parameters(Z)
    logvar = torch.log(var)

    kl = kl_loss(mean, logvar)
    return kl


def mds_loss(X, Y):
    X_dist = torch.cdist(X, X)
    Y_dist = torch.cdist(Y, Y)

    criterion = nn.MSELoss()
    loss = criterion(Y_dist, X_dist)
    return loss


