"""
Bouncing Decayer Optimizer: a physics-inspired optimizer
Adds a decaying oscillatory perturbation to the gradient step.
"""
import math
import torch
from torch.optim import Optimizer

class BouncingDecayerOptimizer(Optimizer):
    """
    Args:
        params (iterable): model parameters
        lr (float): learning rate
        A0 (float): initial amplitude
        lambda_ (float): decay rate
        omega (float): angular frequency
        direction (str): 'gradient' or 'random'
    """
    def __init__(self, params, lr=1e-3, A0=0.1, lambda_=0.01, omega=2.0, direction='gradient'):
        if direction not in ('gradient', 'random'):
            raise ValueError(f"Invalid direction: {direction}")
        defaults = dict(lr=lr, A0=A0, lambda_=lambda_, omega=omega, direction=direction, step=0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None

        for group in self.param_groups:
            lr = group['lr']
            A0 = group['A0']
            lambda_ = group['lambda_']
            omega = group['omega']
            direction = group['direction']
            t = group['step']

            At = A0 * math.exp(-lambda_ * t)
            sin_term = math.sin(omega * t)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if direction == 'gradient':
                    norm = grad.norm()
                    perturb_dir = grad / norm if norm > 0 else torch.zeros_like(grad)
                else:
                    perturb_dir = torch.randn_like(grad)
                    norm = perturb_dir.norm()
                    perturb_dir = perturb_dir / norm if norm > 0 else torch.zeros_like(grad)

                perturb = At * sin_term * perturb_dir

                p.add_( -lr * grad + perturb )

            group['step'] += 1

        return loss
