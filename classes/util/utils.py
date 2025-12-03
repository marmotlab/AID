import math
import numpy as np
import scipy.signal as signal
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


def calculate_position_embedding(node_edges): # Node edges is a list of list of edge index
    num_nodes = len(node_edges)
    A_matrix = np.zeros((num_nodes, num_nodes))
    D_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            # print(f"i is {i}")
            if j in node_edges[i] and i != j:
                A_matrix[i][j] = 1.0
    for i in range(num_nodes):
        D_matrix[i][i] = 1 / np.sqrt(len(node_edges[i]) - 1)
    L = np.eye(num_nodes) - np.matmul(D_matrix, A_matrix, D_matrix)
    eigen_values, eigen_vector = np.linalg.eig(L)
    idx = eigen_values.argsort()
    eigen_values, eigen_vector = eigen_values[idx], np.real(eigen_vector[:, idx])
    eigen_vector = eigen_vector[:, 1:32 + 1]
    return eigen_vector

def calculate_intent_mean_cov(intented_coords):
    estimator = KMeans(n_clusters=1) # 1 cluster for the mean
    data = intented_coords
    estimator.fit(data) # KMeans Intended Coords into 1 cluster
    centroids = estimator.cluster_centers_
    mean = centroids[0]
    cov = np.cov(data, rowvar=False) + np.array([[1 / 28 ** 2, 0], [0, 1 / 28 ** 2]])

    return mean, cov

def calculate_intent_info(num_agents, gaussian_mean, gaussian_cov, agent_id, node_coordinates):
    intent_info = np.zeros((len(node_coordinates), 1))
    for i in range(num_agents):
        if len(gaussian_mean[f"{i}"]) != 0 and i != agent_id:
            Gaussian = multivariate_normal(mean=gaussian_mean[f"{i}"], cov=gaussian_cov[f"{i}"])
            for i in range(len(node_coordinates)):
                X, Y = np.array(node_coordinates[i][0]), np.array(node_coordinates[i][1])
                d = np.dstack([X, Y])
                intent_info[i] += Gaussian.pdf(d)

    if max(intent_info)[0] != 0: # Normalize
        intent_info = intent_info / np.max(intent_info)
    return intent_info

def calculate_intent_difference_KL(cov, cov_before, mean, mean_before):
    intent_difference_KL_1 = 0
    intent_difference_KL_2 = 0
    Gaussian = multivariate_normal(mean=mean, cov=cov)
    Gaussian_before = multivariate_normal(mean=mean_before, cov=cov_before)
    M = 30
    X, Y = np.meshgrid(np.linspace(0, 1, M), np.linspace(0, 1, M))
    d = np.dstack([X, Y])
    Z1 = Gaussian.pdf(d).reshape(M, M)
    Z2 = Gaussian_before.pdf(d).reshape(M, M)
    Z1 = Z1 / np.max(Z1)
    Z2 = Z2 / np.max(Z2)

    for i in range(M):
        for j in range(M):
            if Z1[i][j] < 1e-5:
                Z1[i][j] = 1e-5
            if Z2[i][j] < 1e-5:
                Z2[i][j] = 1e-5
            intent_difference_KL_1 += Z1[i][j] * np.log(Z1[i][j] / Z2[i][j])
            intent_difference_KL_2 += Z2[i][j] * np.log(Z2[i][j] / Z1[i][j])
            # if Z1[i][j] * np.log(Z1[i][j] / Z2[i][j]) > 5:
            #     print(Z1[i][j] * np.log(Z1[i][j] / Z2[i][j]))
            # #     print(Z1[i][j] * np.log(Z1[i][j] / Z2[i][j]), Z2[i][j] * np.log(Z2[i][j] / Z1[i][j]))
            #     print(Z1[i][j], Z2[i][j], np.log(Z1[i][j] / Z2[i][j]), "\n")
    intent_difference_KL = [intent_difference_KL_1, intent_difference_KL_2]

    return intent_difference_KL


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def cal_distance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))
