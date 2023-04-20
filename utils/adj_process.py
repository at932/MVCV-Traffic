import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

def calculate_normalized_laplacian(adj):
    """
    #计算对称归一化拉普拉斯矩阵
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """

    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj, lambda_max=2,undirected=True):
    """
    #L' = 2L/lamda_max - In
    # ‘LM’ : Largest (in magnitude) eigenvalues.
    # 返回1个绝对值最大的特征值与特征向量
    :param adj:
    :param lambda_max:
    :param undirected:是否转换为无向图
    :return:
    """
    # adj = adj + np.eye(adj.shape[0]).astype(np.float32)
    if undirected:
        #转换为无向图
        adj = np.maximum.reduce([adj,adj.T]) #np.maximum.reduce(使用每个位置的最大值创建一个新数组)
    L = calculate_normalized_laplacian(adj)
    if lambda_max is None:
        # 'LM' : Largest (in magnitude) eigenvalues.
        # 返回1个绝对值最大的特征值与特征向量
        lambda_max,_ = linalg.eigsh(L,1,which='LM')
        lambda_max = lambda_max[0]
    #转换为稀疏矩阵
    L = sp.csr_matrix(L)
    M, _ = L.shape #原始矩阵的行数
    I = sp.identity(M,format='csr',dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)

def calculate_random_walk_matrix(adj):
    """
    随机游走矩阵
    D^-1 A
    :param adj:
    :return:
    """
    # adj = adj + np.eye(adj.shape[0]).astype(np.float32)
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv = np.power(d,-1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj).tocoo()
    return random_walk_mx

def calculate_reverse_random_walk_matrix(adj):
    """
    #反向随机游走矩阵
    :param adj:
    :return:
    """
    return calculate_random_walk_matrix(np.transpose(adj))

