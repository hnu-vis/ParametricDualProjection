import math

import torch
from torch.autograd import Function


def torch_norm_pdf(data):
    """
    使用 torch.special.erfcx 计算标准正态分布的 PDF。
    """
    return torch.special.erfcx(data / math.sqrt(2)) / math.sqrt(2 * math.pi)


def torch_norm_cdf(data):
    """
    使用 torch.special.erfcx 计算标准正态分布的 CDF。
    """
    return 0.5 * torch.special.erfcx(-data / math.sqrt(2))


def torch_skewnorm_pdf(data, a, loc, scale):
    """
    计算偏斜分布的概率密度函数 (PDF)。
    """
    scale = scale + 1e-8  # 避免除以零
    y = (data - loc) / scale
    # y = torch.clamp(y, min=-10, max=10)  # 裁剪极端值
    return 2 * torch_norm_pdf(y) * torch_norm_cdf(a * y) / scale


def torch_app_skewnorm_func(data, r, a=-40, loc=0.11, scale=0.13):
    """
    对偏斜分布的 PDF 进行缩放。
    """
    y = torch_skewnorm_pdf(data, a, loc, scale)
    return y * r


class NTXentLoss(Function):
    @staticmethod
    def forward(ctx, probabilities, tau, item_weights):
        """
        前向传播计算 NT-Xent 损失。
        """
        exp_prob = torch.exp(probabilities / tau)
        similarities = exp_prob / torch.sum(exp_prob, dim=1).unsqueeze(1)

        ctx.save_for_backward(similarities, tau, item_weights)
        pos_loss = -torch.log(similarities[:, 0]).mean()

        return pos_loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播，计算 NT-Xent 损失的梯度。
        """
        similarities, t, item_weights = ctx.saved_tensors
        pos_grad_coeff = -((torch.sum(similarities, dim=1) -
                           similarities[:, 0]) / t).unsqueeze(1)
        neg_grad_coeff = similarities[:, 1:] / t
        grad = torch.cat([pos_grad_coeff, neg_grad_coeff],
                         dim=1) * grad_output / similarities.shape[0]

        if item_weights is not None:
            grad *= item_weights.view(-1, 1)

        return grad, None, None


class RefinedNTXentLoss(Function):
    @staticmethod
    def forward(ctx, probabilities, tau, alpha_i, eta, mu, lower_thresh, sigma, item_weight):
        """
        前向传播计算改进的对比损失。
        """
        def nt_xent_grad(data, tau):
            exp_prob = torch.exp(data / tau)
            norm_exp_prob = exp_prob / torch.sum(exp_prob, dim=1).unsqueeze(1)
            gradients = norm_exp_prob[:, 1:] / tau
            return norm_exp_prob, gradients

        similarities, exp_neg_grad_coeff = nt_xent_grad(probabilities, tau)
        skewnorm_prob = torch_skewnorm_pdf(
            probabilities[:, 1:], eta, mu, sigma)
        skewnorm_similarities = skewnorm_prob / \
            torch.sum(skewnorm_prob, dim=1).unsqueeze(1)
        sn_max_val_indices = torch.argmax(skewnorm_similarities, dim=1)
        rows = torch.arange(
            0, skewnorm_similarities.shape[0], 1, device=probabilities.device)
        skewnorm_max_value = skewnorm_similarities[rows, sn_max_val_indices].unsqueeze(
            1)
        ref_exp_value = exp_neg_grad_coeff[rows,
                                           sn_max_val_indices].unsqueeze(1)
        raw_alpha = ref_exp_value / skewnorm_max_value

        ctx.save_for_backward(probabilities, similarities, tau, skewnorm_similarities,
                              mu, lower_thresh, alpha_i, raw_alpha, item_weight)
        pos_loss = -torch.log(similarities[:, 0]).mean()

        return pos_loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播，重新定义负样本对的梯度。
        """
        prob, exp_sims, t, sn_sims, loc, lower_thresh, alpha_i, raw_alpha, item_weights = ctx.saved_tensors
        pos_grad_coeff = -((torch.sum(exp_sims, dim=1) -
                           exp_sims[:, 0]) / t).unsqueeze(1)
        high_thresh = loc

        sn_sims[prob[:, 1:] < lower_thresh] = 0
        sn_sims[prob[:, 1:] >= high_thresh] = 0
        exp_sims[:, 1:][prob[:, 1:] < lower_thresh] = 0

        neg_grad_coeff = exp_sims[:, 1:] / t + alpha_i * sn_sims * raw_alpha
        grad = torch.cat([pos_grad_coeff, neg_grad_coeff],
                         dim=1) * grad_output / exp_sims.shape[0]

        if item_weights is not None:
            grad *= item_weights.view(-1, 1)

        return grad, None, None, None, None, None, None, None


def ntxent_loss(probabilities, tau=0.15, item_weights=None):
    """
    计算 NT-Xent 损失。
    """
    device = probabilities.device
    dtype = probabilities.dtype

    if item_weights is None:
        item_weights = torch.tensor(1.0, device=device, dtype=dtype)
    else:
        item_weights = torch.tensor(item_weights, device=device, dtype=dtype)

    return NTXentLoss.apply(probabilities, torch.tensor(tau, device=device, dtype=dtype), item_weights)


def refined_ntxent_loss(probabilities, tau=0.15, alpha_i=5.0, eta=-40.0, mu=0.11, lower_thresh=0.0, sigma=0.13, item_weights=None):
    device = probabilities.device
    dtype = probabilities.dtype

    if item_weights is None:
        item_weights = torch.tensor(1.0, device=device, dtype=dtype)
    else:
        item_weights = torch.tensor(item_weights, device=device, dtype=dtype)

    return RefinedNTXentLoss.apply(
        probabilities,
        torch.tensor(tau, device=device, dtype=dtype),
        torch.tensor(alpha_i, device=device, dtype=dtype),
        torch.tensor(eta, device=device, dtype=dtype),
        torch.tensor(mu, device=device, dtype=dtype),
        torch.tensor(lower_thresh, device=device, dtype=dtype),
        torch.tensor(sigma, device=device, dtype=dtype),
        item_weights
    )
