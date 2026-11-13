from collections.abc import Callable
from typing import overload

from torch import Tensor
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
)

import torch


def spectral_gd_step(
    w: Tensor,
    grad_w: Tensor,
    lr: float | Tensor,
) -> Tensor:
    
    # svd of gradient
    svd = torch.linalg.svd(
        grad_w, 
        full_matrices=False, 
    )

    # polar factor
    U = svd.U
    Vh = svd.Vh
    polar_k = U @ Vh
    
    # next iterate
    w -= lr * polar_k

    return w


def spectral_gd_cauchy_step(
    w: Tensor,
    grad_w: Tensor,
    hessian: Tensor,
) -> Tensor:
    
    # svd of gradient
    svd = torch.linalg.svd(
        grad_w, 
        full_matrices=False, 
    )

    # polar factor
    U = svd.U
    Vh = svd.Vh
    polar_k = U @ Vh

    a = torch.trace(grad_w.T @ polar_k) 
    b = torch.trace(polar_k @ hessian @ polar_k.T)
    
    lr = a / b
    w -= lr * polar_k

    return w


def frobenius_gd_cauchy_step(
    w: Tensor,
    grad_w: Tensor,
    hessian: Tensor,
) -> Tensor:
    

    a = torch.linalg.norm(grad_w)
    b = torch.trace(
        torch.matmul(
            torch.matmul(grad_w, hessian), 
            grad_w.T,
        )
    )
    
    factor = a ** 2 / b
    w -= factor * grad_w

    return w


class SpectralGD(Optimizer):

    def __init__(
            self,
            params: ParamsT,
            lr: float | Tensor = 1e-3,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
        }
        super().__init__(params, defaults)
        return
    
    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Perform a single optimization step to update parameter.
        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad_w = param.grad
                assert grad_w is not None

                with torch.no_grad():
                    spectral_gd_step(w=param, grad_w=grad_w, lr=lr)

        return loss


class QdfCauchyFrobeniusGD(Optimizer):
    """
    For a quadratic function the hessian is fix 
    therefore we add it in the attributes of the class
    to not recompute it at each step
    """
    def __init__(
            self,
            params: ParamsT,
            hessian: Tensor,
    ) -> None:
        defaults = {
            "hessian": hessian,
        }
        super().__init__(params, defaults)
        return
    
    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Perform a single optimization step to update parameter.
        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            hessian = group["hessian"]
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad_w = param.grad
                assert grad_w is not None

                with torch.no_grad():
                    frobenius_gd_cauchy_step(
                        w=param, 
                        grad_w=grad_w, 
                        hessian=hessian,
                    )

        return loss


class QdfCauchySpectralGD(Optimizer):
    """
    For a quadratic function the hessian is fix 
    therefore we add it in the attributes of the class
    to not recompute it at each step
    """
    def __init__(
            self,
            params: ParamsT,
            hessian: Tensor,
    ) -> None:
        defaults = {
            "hessian": hessian,
        }
        super().__init__(params, defaults)
        return
    
    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Perform a single optimization step to update parameter.
        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            hessian = group["hessian"]
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad_w = param.grad
                assert grad_w is not None

                with torch.no_grad():
                    spectral_gd_cauchy_step(
                        w=param, 
                        grad_w=grad_w, 
                        hessian=hessian,
                    )

        return loss

