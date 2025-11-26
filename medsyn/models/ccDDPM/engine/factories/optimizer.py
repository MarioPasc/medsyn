from typing import Any
import torch.optim as optim

# =============================================================================
# OPTIMIZER FACTORY
# =============================================================================

def create_optimizer(
    model_parameters: Any,
    optimizer_cfg: Any,
) -> Any: # Sorry for Any return type, Lion is a custom optimizer
    """
    Create an optimizer based on configuration.

    Args:
        model_parameters: Model parameters to optimize
        optimizer_cfg: Configuration object with optimizer parameters

    Returns:
        Configured optimizer instance

    Supported optimizer types:
        - adamw: AdamW (default) - Adam with decoupled weight decay
        - adam: Adam - standard Adam optimizer
        - sgd: SGD - stochastic gradient descent with optional momentum
        - rmsprop: RMSprop - adaptive learning rate method
    """
    opt_type = getattr(optimizer_cfg, 'type', 'adamw').lower()
    lr = optimizer_cfg.lr
    wd = optimizer_cfg.wd

    if opt_type == "adamw":
        return optim.AdamW(
            model_parameters,
            lr=lr,
            betas=optimizer_cfg.betas,
            eps=optimizer_cfg.eps,
            weight_decay=wd,
            amsgrad=getattr(optimizer_cfg, 'amsgrad', False),
        )
    elif opt_type == "adam":
        return optim.Adam(
            model_parameters,
            lr=lr,
            betas=optimizer_cfg.betas,
            eps=optimizer_cfg.eps,
            weight_decay=wd,
            amsgrad=getattr(optimizer_cfg, 'amsgrad', False),
        )
    elif opt_type == "sgd":
        return optim.SGD(
            model_parameters,
            lr=lr,
            momentum=getattr(optimizer_cfg, 'momentum', 0.9),
            weight_decay=wd,
            nesterov=getattr(optimizer_cfg, 'nesterov', False),
        )
    elif opt_type == "rmsprop":
        return optim.RMSprop(
            model_parameters,
            lr=lr,
            alpha=getattr(optimizer_cfg, 'alpha', 0.99),
            eps=optimizer_cfg.eps,
            weight_decay=wd,
            momentum=getattr(optimizer_cfg, 'momentum', 0.0),
            centered=getattr(optimizer_cfg, 'centered', False),
        )
    elif opt_type == "lion":
        try:
            # https://github.com/lucidrains/lion-pytorch 
            from lion_pytorch import Lion
        except ImportError:
            raise ImportError("Lion optimizer requires the 'lion-pytorch' package. "
                              "Install it via 'pip install lion-pytorch'.")

        return Lion(
            model_parameters,
            lr=lr,
            weight_decay=wd,
            betas=getattr(optimizer_cfg, 'betas', (0.9, 0.99)),
        )
    else:
        raise ValueError(
            f"Unknown optimizer type: '{opt_type}'. "
            f"Supported types: adamw, adam, sgd, rmsprop, lion"
        )
