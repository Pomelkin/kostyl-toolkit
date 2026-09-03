import math
from collections.abc import Callable
from typing import Any
from typing import override

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT


class AdEMAMix(Optimizer):
    """
    A simple modification of the Adam optimizer with a mixture of two EMAs to better take advantage of past gradients.

    Keeps a fast EMA (``beta1``) that reacts to recent gradients and a slow EMA (``beta3``) that retains
    older ones; the update uses ``exp_avg + alpha * exp_avg_slow`` normalized by the usual Adam denominator.
    ``T_alpha_beta3`` optionally warms up ``alpha`` and ``beta3`` over the first iterations to stabilize early training.

    Reference: https://arxiv.org/abs/2409.03137
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        alpha: float = 5.0,
        T_alpha_beta3: float | None = None,
    ) -> None:
        """
        Configure the AdEMAMix optimizer.

        Args:
            params: Parameters to optimize or dicts defining param groups.
            lr: Learning rate.
            betas: Coefficients ``(beta1, beta2, beta3)`` for the fast EMA, the squared-gradient EMA
                and the slow EMA respectively.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay coefficient.
            alpha: Weight of the slow EMA in the update.
            T_alpha_beta3: Optional number of iterations over which ``alpha`` and ``beta3`` are warmed up
                to their target values. If None, both are used at full strength from the first step.
        """
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not eps >= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if len(betas) != 3 or any(b < 0.0 or b >= 1.0 for b in betas):
            raise ValueError(f"Invalid beta parameters: {betas}")

        if not weight_decay >= 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "alpha": alpha,
            "T_alpha_beta3": T_alpha_beta3,
        }
        super().__init__(params, defaults)
        return

    @override
    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        return

    @override
    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:  # ty: ignore[invalid-method-override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            exp_avg_slow: list[torch.Tensor] = []
            state_steps: list[int] = []

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError("AdEMAMix does not support sparse gradients")
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Slow exponential moving average
                        state["exp_avg_slow"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    exp_avg_slow.append(state["exp_avg_slow"])
                    state["step"] += 1
                    state_steps.append(state["step"])

            beta1, beta2, beta3 = group["betas"]
            alpha = group["alpha"]
            T_alpha_beta3 = group["T_alpha_beta3"]

            self._update_adamemix(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                exp_avg_slow,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                alpha=alpha,
                T_alpha_beta3=T_alpha_beta3,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )

        return loss

    def _update_adamemix(
        self,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        exp_avg_slow: list[torch.Tensor],
        state_steps: list[int],
        beta1: float,
        beta2: float,
        beta3: float,
        alpha: float,
        T_alpha_beta3: float | None,
        lr: float,
        weight_decay: float,
        eps: float,
    ) -> None:

        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            exp_avg_slow_i = exp_avg_slow[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            if T_alpha_beta3 is not None:
                alpha_t = min(step * alpha / T_alpha_beta3, alpha)
                beta3_t = min(
                    math.exp(
                        math.log(beta1)
                        * math.log(beta3)
                        / (
                            (1 - step / T_alpha_beta3) * math.log(beta3)
                            + (step / T_alpha_beta3) * math.log(beta1)
                        )
                    ),
                    beta3,
                )
            else:
                alpha_t = alpha
                beta3_t = beta3

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            exp_avg_slow_i.mul_(beta3_t).add_(grad, alpha=1 - beta3_t)

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            if weight_decay != 0:
                param.add_(param, alpha=-weight_decay * lr)

            param.addcdiv_(exp_avg + alpha_t * exp_avg_slow_i, denom, value=-step_size)
        return
