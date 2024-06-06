import random
from types import NoneType

import numpy as np
from sklearn.mixture import GaussianMixture

from utils import build_dataset_for_AAS

eps = 1e-6


def gauss(data, mu, sigma2):
    gaussian = ((1.0 / (2 * np.pi * np.sqrt(np.linalg.det(sigma2)))) *
                np.exp((-1 / 2.0) * ((data - mu) @ np.linalg.inv(sigma2) @ np.transpose(data - mu))))
    if gaussian < eps:
        gaussian = eps
    return gaussian


def build_dataset_with_extra_features(path):  # most generic EM variant
    dataset, dataset_len = build_dataset_for_AAS('')
    initial = 16
    extra = 16
    no_dist = 2
    max_iter = 100
    data = np.array([instance[0] + instance[0] for instance in dataset])
    # sample = np.random.random_sample((no_dist, initial + extra))
    # means = np.array([sample for j in range(no_dist)])
    means = np.array([np.mean([data[:, j] for j in range(initial + extra)], axis=1) for _ in range(no_dist)])
    # means = np.ones((no_dist, initial + extra))
    sigma2 = np.array([np.diag(np.var([data[:, j] for j in range(initial + extra)], axis=1) * 0.9)
                       for _ in range(no_dist)])
    # sigma2 = np.array([np.identity(initial + extra) *
    #                    data[random.randint(0, len(data) - 1)][random.randint(0, initial + extra - 1)]
    #                    for _ in range(no_dist)])
    # sample = np.random.random_sample((no_dist, initial + extra))
    # sigma2 = np.array([np.identity(initial + extra) * 1e-6 for _ in range(no_dist)])
    # sigma2 = np.array([np.diag(sample[j]) for j in range(no_dist)])
    probs = np.array([1.0 / no_dist for _ in range(no_dist)])
    z = np.zeros((len(data), no_dist))
    # for i in range(len(data)):
    #     k = random.randint(0, no_dist - 1)
    #     data[i][initial:] = np.random.multivariate_normal(means[k][initial:], sigma2[k][initial:, initial:])
    for it in range(max_iter):
        # E: update z
        for i in range(len(data)):
            suma = sum([probs[j] * gauss(data[i], means[j], sigma2[j]) for j in range(no_dist)])
            for j in range(no_dist):
                z[i][j] = probs[j] * gauss(data[i], means[j], sigma2[j]) / suma
        print(z[0])
        # M: update parameters + missing data
        for j in range(no_dist):
            probs[j] = 1 / len(data) * sum([z[i][j] for i in range(len(data))])
            means[j] = (np.sum([z[i][j] * data[i] for i in range(len(data))], axis=0)
                        / sum([z[i][j] for i in range(len(data))]))
            # sigma2[j] = np.sum([z[i][j] *
            #                     np.matmul((data[i] - means[j]).reshape(initial + extra, 1),
            #                               (data[i] - means[j]).T.reshape(1, initial + extra))
            #                     for i in range(len(data))], axis=0) / sum([z[i][j] for i in range(len(data))])
            sigma = (sum([((data[i] - means[j]) @ (data[i] - means[j])) * z[i][j] for i in range(len(data))])
                     / (sum([z[i][j] for i in range(len(data))]) * (initial + extra)))
            sigma2[j] = np.diag([sigma for _ in range(initial + extra)])

        # for index in range(initial, initial + extra):
        #     for i in range(len(data)):
        #         data[i][index] =

        print(f"Iteration {it} finished.")
    dataset_transformed = [[instance.tolist(), dataset[index][1]] for index, instance in enumerate(data)]
    return dataset_transformed, dataset_len


print(build_dataset_with_extra_features(''))
