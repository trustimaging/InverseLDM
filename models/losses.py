import torch
import torch.nn.functional as F


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
