import numpy as np
from sklearn.neighbors import NearestNeighbors


def trustworthiness(X_high, X_low, k=30, distance_matrix_high=None):
    N = X_high.shape[0]

    # 使用自定义距离矩阵或默认欧氏距离
    if distance_matrix_high is not None:
        nbrs_high = NearestNeighbors(
            n_neighbors=N, metric='precomputed').fit(distance_matrix_high)
    else:
        nbrs_high = NearestNeighbors(n_neighbors=N).fit(X_high)

    nbrs_low = NearestNeighbors(n_neighbors=k+1).fit(X_low)

    # 获取高维数据中所有样本的最近邻排名
    if distance_matrix_high is not None:
        _, high_indices = nbrs_high.kneighbors(distance_matrix_high)
    else:
        _, high_indices = nbrs_high.kneighbors(X_high)

    # 获取低维数据中所有样本的最近邻
    _, low_indices = nbrs_low.kneighbors(X_low)
    low_indices = low_indices[:, 1:]  # 排除自身

    # 计算 T30
    trust = 0
    for i in range(N):
        for j in low_indices[i]:
            if j not in high_indices[i, :k+1]:  # 检查是否在高维的 k 最近邻中
                rank = np.where(high_indices[i] == j)[0][0]  # 获取排名
                trust += max(0, rank - k)  # 只计算超出 k 的部分

    trust = 1 - (2 / (N * k * (2 * N - 3 * k - 1))) * trust
    return trust


def continuity(X_high, X_low, k=30, distance_matrix_high=None):
    N = X_high.shape[0]

    # 使用自定义距离矩阵或默认欧氏距离
    if distance_matrix_high is not None:
        nbrs_high = NearestNeighbors(
            n_neighbors=k+1, metric='precomputed').fit(distance_matrix_high)
    else:
        nbrs_high = NearestNeighbors(n_neighbors=k+1).fit(X_high)

    # 使用低维数据计算最近邻（默认欧氏距离）
    nbrs_low = NearestNeighbors(n_neighbors=N).fit(X_low)

    # 获取高维数据中所有样本的最近邻
    if distance_matrix_high is not None:
        _, high_indices = nbrs_high.kneighbors(distance_matrix_high)
    else:
        _, high_indices = nbrs_high.kneighbors(X_high)
    high_indices = high_indices[:, 1:]  # 排除自身

    # 获取低维数据中所有样本的最近邻排名
    _, low_indices = nbrs_low.kneighbors(X_low)

    # 计算 C30
    cont = 0
    for i in range(N):
        for j in high_indices[i]:
            if j not in low_indices[i, :k+1]:  # 检查是否在低维的 k 最近邻中
                rank = np.where(low_indices[i] == j)[0][0]  # 获取排名
                cont += max(0, rank - k)  # 只计算超出 k 的部分

    cont = 1 - (2 / (N * k * (2 * N - 3 * k - 1))) * cont
    return cont
