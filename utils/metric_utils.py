import numpy as np
from sklearn.neighbors import NearestNeighbors


def trustworthiness(X_high, X_low, k=30, distance_matrix_high=None):
    """Computes the trustworthiness metric for dimensionality reduction evaluation.
    
    Measures how well the local structure of high-dimensional data is preserved in 
    low-dimensional embeddings. A value of 1 indicates perfect preservation of 
    k-nearest neighbors, while lower values indicate more neighbor violations.
    
    Args:
        X_high: High-dimensional data array of shape (n_samples, n_features)
        X_low: Low-dimensional embedding array of shape (n_samples, n_components)
        k: Number of nearest neighbors to consider (default: 30)
        distance_matrix_high: Precomputed distance matrix for high-dimensional data.
                             If provided, uses this instead of computing from X_high.
    
    Returns:
        float: Trustworthiness score between 0 and 1, where higher is better.
    
    Example:
        >>> X_high = np.random.rand(100, 50)
        >>> X_low = np.random.rand(100, 2)
        >>> score = trustworthiness(X_high, X_low, k=15)
    """
    N = X_high.shape[0]

    if distance_matrix_high is not None:
        neighbors_high = NearestNeighbors(
            n_neighbors=N, metric='precomputed').fit(distance_matrix_high)
    else:
        neighbors_high = NearestNeighbors(n_neighbors=N).fit(X_high)

    neighbors_low = NearestNeighbors(n_neighbors=k+1).fit(X_low)

    if distance_matrix_high is not None:
        _, high_indices = neighbors_high.kneighbors(distance_matrix_high)
    else:
        _, high_indices = neighbors_high.kneighbors(X_high)

    _, low_indices = neighbors_low.kneighbors(X_low)
    low_indices = low_indices[:, 1:]

    trust = 0
    for i in range(N):
        for j in low_indices[i]:
            if j not in high_indices[i, :k+1]:
                rank = np.where(high_indices[i] == j)[0][0]
                trust += max(0, rank - k)

    trust = 1 - (2 / (N * k * (2 * N - 3 * k - 1))) * trust
    return trust


def continuity(X_high, X_low, k=30, distance_matrix_high=None):
    """Computes the continuity metric for dimensionality reduction evaluation.
    
    Measures how well the original neighbors are preserved in the low-dimensional 
    embedding. Complementary to trustworthiness, with 1 indicating perfect 
    preservation of original neighborhoods.
    
    Args:
        X_high: High-dimensional data array of shape (n_samples, n_features)
        X_low: Low-dimensional embedding array of shape (n_samples, n_components)
        k: Number of nearest neighbors to consider (default: 30)
        distance_matrix_high: Precomputed distance matrix for high-dimensional data.
                             If provided, uses this instead of computing from X_high.
    
    Returns:
        float: Continuity score between 0 and 1, where higher is better.
    
    Example:
        >>> X_high = np.random.rand(100, 50)
        >>> X_low = np.random.rand(100, 2)
        >>> score = continuity(X_high, X_low, k=15)
    """
    N = X_high.shape[0]

    if distance_matrix_high is not None:
        nbrs_high = NearestNeighbors(
            n_neighbors=k+1, metric='precomputed').fit(distance_matrix_high)
    else:
        nbrs_high = NearestNeighbors(n_neighbors=k+1).fit(X_high)

    nbrs_low = NearestNeighbors(n_neighbors=N).fit(X_low)

    if distance_matrix_high is not None:
        _, high_indices = nbrs_high.kneighbors(distance_matrix_high)
    else:
        _, high_indices = nbrs_high.kneighbors(X_high)
    high_indices = high_indices[:, 1:]

    _, low_indices = nbrs_low.kneighbors(X_low)

    cont = 0
    for i in range(N):
        for j in high_indices[i]:
            if j not in low_indices[i, :k+1]:
                rank = np.where(low_indices[i] == j)[0][0]
                cont += max(0, rank - k)

    cont = 1 - (2 / (N * k * (2 * N - 3 * k - 1))) * cont
    return cont
