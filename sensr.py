import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression


def calc_sensitive_directions(x_no_sensitive, sensitive_group):
    """
    x_no_sensitive: x without the sensitive features
    sensitive_group: sensitive features (can be more than one column)
    """
    sensitive_directions = []
    if np.ndim(sensitive_group) == 1:
        sensitive_group = sensitive_group.reshape(1, -1)
    for y_protected in sensitive_group:
        lr = LogisticRegression(
            solver="liblinear", fit_intercept=True, penalty="l1"
        )
        lr.fit(x_no_sensitive, y_protected)
        sensitive_directions.append(lr.coef_.flatten())

    sensitive_directions = np.array(sensitive_directions)
    return sensitive_directions


def normalize_sensitive_directions(basis):
    proj = np.linalg.inv(np.matmul(basis.T, basis))
    proj = np.matmul(basis, proj)
    proj = np.matmul(proj, basis.T)
    return proj


def compl_svd_projector(sensitive_directions, svd=-1):
    if svd > 0:
        tSVD = TruncatedSVD(n_components=svd)
        tSVD.fit(sensitive_directions)
        basis = tSVD.components_.T
    else:
        basis = sensitive_directions.T

    proj = normalize_sensitive_directions(basis)
    proj_compl = np.eye(proj.shape[0]) - proj
    return proj_compl


def fair_dist(proj_compl):
    proj_compl_ = torch.FloatTensor(proj_compl)
    return lambda x, y: torch.sum(
        torch.square(torch.matmul(x - y, proj_compl_)), dim=1
    )


def unprotected_direction(x, sensetive_directions):
    x = x - x @ sensetive_directions
    return x


def sample_perturbation(
    model,
    x,
    y,
    sensitive_directions,
    regularizer=100,
    learning_rate=5e-2,
    num_steps=200,
):
    x_start = x.clone().detach()
    x_ = x.clone()
    x_.requires_grad = True
    proj_compl = compl_svd_projector(sensitive_directions)
    fair_loss = fair_dist(proj_compl)
    for i in range(num_steps):
        prob = model(x_)
        perturb = fair_loss(x_, x_start)
        loss = (
            nn.CrossEntropyLoss()(prob, y)
            - regularizer * torch.linalg.norm(perturb) ** 2
        )
        gradient = torch.autograd.grad(loss, x_)
        x_ = x_ + learning_rate * gradient[0]
        x_start = x_.clone().detach()
    return x_.detach()
