import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from torch.autograd import Variable
from tqdm import tqdm

from models import ModelWrapper


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
    x_start = x.clone().detach() + min(1e-2, learning_rate) * torch.rand_like(
        x
    )
    x_ = x.clone()
    x_.requires_grad = True
    proj_compl = compl_svd_projector(sensitive_directions)
    fair_metric = fair_dist(proj_compl)
    for i in range(num_steps):
        prob = model(x_)
        perturb = fair_metric(x_, x_start)
        fair_loss = (torch.linalg.norm(perturb) ** 2).clamp_(0, 1e13)
        loss = nn.CrossEntropyLoss()(prob, y) - regularizer * fair_loss
        gradient = torch.autograd.grad(loss, x_, retain_graph=False)
        x_ = x_ + learning_rate * gradient[0]
    return x_.detach()


def train_fair_model(
    model,
    train_loader,
    sensitive_directions,
    epochs,
    fair_epochs,
    n_features,
    eps=0.1,
    lmbd_init=2.0,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    optimizer_clf = optim.Adam(model.parameters(), lr=0.005)
    loss_classifier = torch.nn.CrossEntropyLoss()
    sensitive_directions_ = normalize_sensitive_directions(
        sensitive_directions.T
    )
    proj_cmpl = compl_svd_projector(sensitive_directions)
    sensitive_directions_ = torch.FloatTensor(sensitive_directions_)
    fair_loss = fair_dist(proj_cmpl)

    start_fair_step = epochs // 2

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        for x, y, _ in train_loader:
            lmbd = lmbd_init
            x, y = x.to(device), y.to(device)
            x.requires_grad = False
            if epoch < start_fair_step:
                clf_output = model(x)
                clf_loss = loss_classifier(clf_output, y)
                optimizer_clf.zero_grad()
                clf_loss.backward()
                optimizer_clf.step()
            else:
                batch_size = x.size(0)

                full_adv_weights = eps * torch.rand(
                    (batch_size, n_features),
                    device=device,
                    requires_grad=False,
                )
                full_adv_weights.requires_grad = True
                
                d_sens = sensitive_directions_.size(0)
                adv_weights = eps * torch.rand(
                    (batch_size, d_sens), device=device, requires_grad=False
                )
                adv_weights.requires_grad = True

                optim_adv = optim.Adam([adv_weights], lr=0.001)
                optim_full_adv = optim.Adam([full_adv_weights], lr=0.001)
                                
                x_fair = (
                    x
                    + torch.matmul(adv_weights, sensitive_directions_)
                    + full_adv_weights
                )

                clf_output = model(x_fair)
                clf_loss = loss_classifier(clf_output, y)

                optim_adv.zero_grad()
                loss_adv = -clf_loss
                loss_adv.backward(retain_graph=True)
                optim_adv.step()

                for _ in range(fair_epochs):
                    clf_output = model(x_fair)
                    clf_loss = loss_classifier(clf_output, y)
                    distance = fair_loss(x, x_fair)
                    tot_loss = clf_loss - lmbd * (torch.linalg.norm(distance) ** 2)
                    optim_full_adv.zero_grad()
                    loss_adv = -tot_loss
                    loss_adv.backward(retain_graph=True)
                    optim_full_adv.step()

                    lmbd = max(
                        0,
                        lmbd
                        + max(distance.mean().detach(), eps)
                        / min(distance.mean().detach(), eps)
                        * (distance.mean().detach() - eps),
                    )
                
                clf_output = model(x_fair)
                clf_loss = loss_classifier(clf_output, y)
                optimizer_clf.zero_grad()
                clf_loss.backward()
                optimizer_clf.step()