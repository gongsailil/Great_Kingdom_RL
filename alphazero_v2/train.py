"""Tiny in-memory policy/value training and checkpoint utilities."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import AlphaZeroConfig
from .network import PolicyValueNetwork


def _stack_examples(examples, device):
    states = torch.from_numpy(np.stack([item.state for item in examples])).to(device)
    policies = torch.from_numpy(np.stack([item.policy for item in examples])).to(device)
    values = torch.tensor(
        [item.value for item in examples],
        dtype=torch.float32,
        device=device,
    )
    return states, policies, values


def loss_components(network, states, target_policies, target_values):
    logits, values = network(states)
    policy_loss = -(target_policies * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(values, target_values)
    total_loss = policy_loss + value_loss
    return policy_loss, value_loss, total_loss


def evaluate_losses(network, examples, device):
    states, policies, values = _stack_examples(examples, device)
    was_training = network.training
    network.eval()
    with torch.no_grad():
        losses = loss_components(network, states, policies, values)
    if was_training:
        network.train()
    return {
        "policy": float(losses[0].item()),
        "value": float(losses[1].item()),
        "total": float(losses[2].item()),
    }


def train_on_examples(network, examples, config, device, optimizer=None):
    if not examples:
        raise ValueError("cannot train without self-play examples")
    device = torch.device(device)
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    initial_losses = evaluate_losses(network, examples, device)
    states, policies, values = _stack_examples(examples, device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    for _ in range(config.training_epochs):
        permutation = torch.randperm(len(examples), generator=generator)
        network.train()
        for start in range(0, len(examples), config.batch_size):
            indices = permutation[start : start + config.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = loss_components(
                network,
                states[indices],
                policies[indices],
                values[indices],
            )
            losses[2].backward()
            optimizer.step()

    final_losses = evaluate_losses(network, examples, device)
    return optimizer, initial_losses, final_losses


def save_checkpoint(path, network, optimizer, config, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "network_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
            "training_metadata": metadata,
        },
        path,
    )


def load_checkpoint(path, device):
    device = torch.device(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    config = AlphaZeroConfig.from_dict(payload["config"])
    network = PolicyValueNetwork(
        channels=config.channels,
        residual_blocks=config.residual_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    network.load_state_dict(payload["network_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return network, optimizer, config, payload["training_metadata"]
