"""Fit one held-out scalar temperature for the V5 Bernoulli value logit."""

import numpy as np
import torch
from torch.nn import functional as F


def _metrics(logits, targets, weights, temperature):
    scaled = logits / float(temperature)
    normalized = weights / weights.sum()
    nll = (F.binary_cross_entropy_with_logits(scaled, targets, reduction="none") * normalized).sum()
    probabilities = torch.sigmoid(scaled)
    brier = (torch.square(probabilities - targets) * normalized).sum()
    return {"nll": float(nll.item()), "brier": float(brier.item())}


def validation_value_logits(network, view, device, batch_size=4096):
    states, targets, weights = view.validation_arrays()
    outputs = []
    was_training = network.training
    network.eval()
    with torch.no_grad():
        for start in range(0, len(states), int(batch_size)):
            inputs = torch.from_numpy(states[start:start + batch_size]).to(device)
            outputs.append(network(inputs)[1].detach().cpu())
    if was_training:
        network.train()
    return (torch.cat(outputs).float(), torch.from_numpy(targets).float(),
            torch.from_numpy(weights).float())


def fit_value_temperature(network, view, device):
    logits, targets, weights = validation_value_logits(network, view, device)
    before = _metrics(logits, targets, weights, 1.0)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    logits64, targets64, weights64 = logits.double(), targets.double(), weights.double()
    normalized = weights64 / weights64.sum()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.25, max_iter=100,
                                  tolerance_grad=1e-10, tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(torch.clamp(log_temperature, -4.0, 4.0))
        loss = (F.binary_cross_entropy_with_logits(
            logits64 / temperature, targets64, reduction="none") * normalized).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(np.clip(np.exp(log_temperature.detach().item()), np.exp(-4), np.exp(4)))
    after = _metrics(logits, targets, weights, temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("non-finite calibration temperature")
    # Numerical optimization may land a few ulps worse; retain unscaled logits then.
    if after["nll"] > before["nll"] + 1e-8:
        temperature, after = 1.0, before
    return {"temperature": temperature, "groups": len(targets),
            "raw_weight": int(weights.sum().item()),
            "before_nll": before["nll"], "after_nll": after["nll"],
            "before_brier": before["brier"], "after_brier": after["brier"]}
