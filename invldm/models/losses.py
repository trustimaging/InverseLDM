import torch
import torch.nn.functional as F
from awloss import AWLoss
from ..utils.utils import scale2range
from functools import partial


def divergence(args, mean, log_var):
    if args.model.div_loss.lower() == "kl":
        return 0.5 * torch.sum(torch.pow(mean, 2)
                               + log_var.exp() - 1.0
                               - log_var, dim=[1, 2, 3])
    else:
        raise NotImplementedError(
            "Currently only supporting the KL divergence loss"
        )


def perceptual(args, input, recon):
    if not args.model.perceptual_loss:
        return torch.tensor([0.])
    elif args.model.perceptual_loss == "wiener":
        epsilon = args.params.__dict__.get("wiener_epsilon", 250.)
        alpha = args.params.__dict__.get("wiener_alpha", -0.2)
        beta = args.params.__dict__.get("wiener_beta", 1.5)
        filter_scale = args.params.__dict__.get("wiener_filter_scale", 2)

        awloss = AWLoss(filter_dim=2, method="fft", reduction="mean", store_filters=False,
                        epsilon=epsilon, filter_scale=filter_scale,
                        penalty_function=partial(laplacian2D, alpha=alpha, beta=beta))
        # print(awloss.__dict__)
        return awloss(input, recon)
    else:
        raise NotImplementedError(
            "Currently not supporting perceptual losses"
        )


def reconstruction(args, input, recon):
    loss_type = None
    try:
        loss_type = args.model.recon_loss.lower()
    except AttributeError:
        loss_type = args.model.loss.lower()

    if loss_type == "l2":
        return F.mse_loss(input, recon)
    elif loss_type == "l1":
        return F.l1_loss(input, recon)
    else:
        raise NotImplementedError(
            "Currently only supporting the L1 and L2 reconstruction losses"
        )


def laplacian2D(mesh, alpha=-0.2, beta=1.5):
    xx, yy = mesh[:,:,0], mesh[:,:,1]
    x = torch.sqrt(xx**2 + yy**2) 
    T = 1 - torch.exp(-torch.abs(x) ** alpha) ** beta
    T = scale2range(T, [0.05, 1.])
    return T